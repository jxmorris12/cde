from typing import Tuple

import gc
import glob
import os

import datasets
import pyarrow as pa
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer
import torch

from cde.lib import (
    gather,
    get_cde_cache_dir, 
    get_rank,
    get_world_size,
    md5_hash_kwargs, 
    print0, 
    tqdm_if_main_worker
)

def save_embeddings(embeddings: torch.Tensor, file_path: str):
    table = pa.Table.from_arrays([pa.array(embeddings)], names=['embeddings'])
    # Write to Parquet file
    pq.write_table(table, file_path)
    print(f"Saved {len(embeddings)} embeddings to {file_path}.")


MAX_NUM_EMBEDDINGS_PER_SHARD = 10_000_000

class TrainerNegativeFilterMixin:
    """Filters hard negatives based on another pre-trained model."""
    _document_embedding_index = None
    _query_embedding_index = None
    @property
    def _filename_dir(self) -> str:
        dirname = md5_hash_kwargs(
            dataset_fingerprint=self.train_dataset._fingerprint,
            hn_filter_model=self.args.hn_filter_model,
            seq_length=self.model.config.max_seq_length,
        )
        return os.path.join(get_cde_cache_dir(), 'hn_embeddings', dirname)
    
    def _filename_doc_index(self, shard: int) -> str:
        return os.path.join(self._filename_dir, f"corpus_shard_{shard}.parquet")

    def _filename_query_index(self, shard: int) -> str:
        return os.path.join(self._filename_dir, f"query_shard_{shard}.parquet")
    
    @property
    def _filename_finished_sentinel(self) -> str:
        return os.path.join(self._filename_dir, "finished")
    
    def _init_hn_filter_model(self):
        if self._hn_filter_model is None:
            if self.args.hn_filter_model == "nomic":
                self._hn_filter_model = SentenceTransformer(
                    "nomic-ai/nomic-embed-text-v1", 
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    trust_remote_code=True,
                )
                self._hn_filter_model.to(self.args.device)
            elif self.args.hn_filter_model == "stella":
                self._hn_filter_model = SentenceTransformer(
                    "dunzhang/stella_en_1.5B_v5", 
                    trust_remote_code=True, 
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    model_kwargs={
                        # "torch_dtype": torch.bfloat16,
                        # "attn_implementation": "flash_attention_2",
                    }
                )
                print0(f"[trainer_hn_filter] Loaded model stella_en_1.5B_v5 and set max_seq_length to {self.model.config.max_seq_length}.")
                self._hn_filter_model.max_seq_length = self.model.config.max_seq_length
            elif self.args.hn_filter_model == "sbert":
                self._hn_filter_model = SentenceTransformer(
                    "all-MiniLM-L6-v2",
                    device="cuda" if torch.cuda.is_available() else "cpu",
                )
                
                print0(f"[trainer_hn_filter] Loaded model all-MiniLM-L6-v2 and set max_seq_length to {self.model.config.max_seq_length}.")
                self._hn_filter_model.max_seq_length = self.model.config.max_seq_length
            elif self.args.hn_filter_model == "sentence_t5":
                self._hn_filter_model = SentenceTransformer(
                    "sentence-transformers/sentence-t5-base",
                    device="cuda" if torch.cuda.is_available() else "cpu",
                )
                print0(f"[trainer_hn_filter] Loaded model sentence-transformers/sentence-t5-base and set max_seq_length to {self.model.config.max_seq_length}.")
                self._hn_filter_model.max_seq_length = self.model.config.max_seq_length
            else:
                raise ValueError(f"Unknown hn_filter_model: {self.args.hn_filter_model}")

        if not self.args.hn_filter_precompute_vectors:
            return
        
        print0(f"[trainer_hn_filter] Checking for sentinel file {self._filename_finished_sentinel}.")
        if not os.path.exists(self._filename_finished_sentinel):
            print0(f"[trainer_hn_filter] Precomputing hn index for {self.args.hn_filter_model} (max_seq_length={self.model.config.max_seq_length}, batch_size={self._inference_batch_size})...")
            print0(f"[trainer_hn_filter] shard size = {MAX_NUM_EMBEDDINGS_PER_SHARD}.")
            self._precompute_hn_index()
        
        num_shards = len(glob.glob(os.path.join(self._filename_dir, "corpus_shard_*.parquet")))
        document_embedding_index = datasets.load_dataset("parquet", data_files=[self._filename_doc_index(shard) for shard in range(num_shards)])["train"]
        document_embedding_index = document_embedding_index.rename_column("embeddings", "document_embedding")
        
        query_embedding_index = datasets.load_dataset("parquet", data_files=[self._filename_query_index(shard) for shard in range(num_shards)])["train"]
        query_embedding_index = query_embedding_index.rename_column("embeddings", "query_embedding")

        og_fingerprint = self.train_dataset._fingerprint
        self.train_dataset.dataset = datasets.concatenate_datasets([self.train_dataset.dataset, document_embedding_index, query_embedding_index], axis=1)
        self.train_dataset._fingerprint = og_fingerprint
    
    @property
    def _inference_batch_size(self) -> int:
        return min(self.args.max_batch_size_fits_in_memory, self.args.per_device_train_batch_size)
        
    def _precompute_hn_index(self):
        # Precompute vectors
        batch_size = 4096

        train_dataset_length = (len(self.train_dataset) // (batch_size * get_world_size())) * batch_size * get_world_size()
        print0(f"Precomputing embeddings for {len(self.train_dataset)} samples: {train_dataset_length} across {get_world_size()} workers and {len(self.train_dataset) - train_dataset_length} on self.")
        dataset1 = self.train_dataset.dataset).select(range(0, train_dataset_length)
        train_dataloader_1 = torch.utils.data.DataLoader(
            dataset1, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=self.data_collator
        )
        # Do first train dataloader across world
        world_size, rank = get_world_size(), get_rank()
        all_query_embeddings, all_doc_embeddings = [], []
        shard_idx = 0
        for j, batch in tqdm_if_main_worker(enumerate(train_dataloader_1), total=len(train_dataloader_1), desc="Precomputing embeddings"):
            if (j % world_size) != rank:
                # poor man's distributed sampler
                continue
            
            query_embeddings, doc_embeddings = self._get_embeddings(
                query_inputs={ "text": batch["query"], },
                document_inputs={ "text": batch["document"], },
            )
            batch_query_embeddings = gather(query_embeddings)
            batch_doc_embeddings = gather(doc_embeddings)
            if self._is_main_worker:
                all_query_embeddings.extend(batch_query_embeddings.cpu().to(torch.float16).numpy().tolist())
                all_doc_embeddings.extend(batch_doc_embeddings.cpu().to(torch.float16).numpy().tolist())
            
            if len(all_query_embeddings) > MAX_NUM_EMBEDDINGS_PER_SHARD:
                # Save the index to disk
                if self._is_main_worker:
                    os.makedirs(self._filename_dir, exist_ok=True)
                    save_embeddings(all_query_embeddings, self._filename_query_index(shard=shard_idx))
                    save_embeddings(all_doc_embeddings, self._filename_doc_index(shard=shard_idx))
                
                gc.collect()
                torch.cuda.empty_cache()
                print0(f"[trainer_hn_filter] Saved {len(all_query_embeddings)} query+doc embeddings to disk (shard={shard_idx}).")
                shard_idx += 1
                all_query_embeddings, all_doc_embeddings = [], []
        # Compute last samples on main worker to even it out
        dataset2 = self.train_dataset.dataset).select(range(train_dataset_length, len(self.train_dataset))
        train_dataloader_2 = torch.utils.data.DataLoader(
            dataset2, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=self.data_collator
        )
        for batch in tqdm_if_main_worker(train_dataloader_2, total=len(train_dataloader_2), desc="Precomputing embeddings (remainder)"):
            batch_query_embeddings, batch_doc_embeddings = self._get_embeddings(
                query_inputs={ "text": batch["query"] },
                document_inputs={ "text": batch["document"] },
            )
            if self._is_main_worker:
                all_query_embeddings.extend(batch_query_embeddings.cpu().to(torch.float16).numpy().tolist())
                all_doc_embeddings.extend(batch_doc_embeddings.cpu().to(torch.float16).numpy().tolist())
        # Save the index to disk
        if self._is_main_worker:
            os.makedirs(self._filename_dir, exist_ok=True)
            save_embeddings(all_query_embeddings, self._filename_query_index(shard=shard_idx))
            save_embeddings(all_doc_embeddings, self._filename_doc_index(shard=shard_idx))
            # All done
            open(self._filename_finished_sentinel, 'a').close()

        try:
            torch.distributed.barrier()    
        except ValueError:
            # ValueError: Default process group has not been initialized, please make sure to call init_process_group.
            pass
        print0("[trainer_hn_filter] Saved index to disk.")
    
    def _get_embeddings_sentence_transformers(self, query_inputs, document_inputs) -> Tuple[torch.Tensor, torch.Tensor]:
        model = self._hn_filter_model
        queries = query_inputs["text"]
        docs = document_inputs["text"]
        with torch.no_grad():
            query_embeddings = model.encode(
                queries,
                batch_size=(self._inference_batch_size * 2),
                convert_to_tensor=True
            )
            doc_embeddings = model.encode(
                docs,    
                batch_size=(self._inference_batch_size * 2),
                convert_to_tensor=True
            )
        return query_embeddings, doc_embeddings
            
    def _get_embeddings_nomic(self, query_inputs, document_inputs) -> Tuple[torch.Tensor, torch.Tensor]:
        from cde.lib.tensor import forward_batched
    
        queries = query_inputs["text"]
        docs = document_inputs["text"]
        
        model = self._hn_filter_model
        with torch.no_grad():
            query_embeddings = model.encode(
                [f"search_query: " + q for q in queries],    
                batch_size=(self._inference_batch_size),
                convert_to_tensor=True
            )
            doc_embeddings = model.encode(
                [f"search_document: " + d for d in docs],    
                batch_size=(self._inference_batch_size),
                convert_to_tensor=True
            )

        query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
        doc_embeddings = torch.nn.functional.normalize(doc_embeddings, p=2, dim=1)
        return query_embeddings, doc_embeddings
    
    def _get_embeddings_stella(self, query_inputs, document_inputs) -> Tuple[torch.Tensor, torch.Tensor]:
        # https://huggingface.co/dunzhang/stella_en_1.5B_v5
        query_prompt_name = "s2p_query"
    
        queries = query_inputs["text"]
        docs = document_inputs["text"]
        
        model = self._hn_filter_model
        with torch.no_grad():
            query_embeddings = model.encode(
                queries,
                prompt_name=query_prompt_name, 
                batch_size=(self._inference_batch_size * 2),
                convert_to_tensor=True
            )
            doc_embeddings = model.encode(
                docs,    
                batch_size=(self._inference_batch_size * 2),
                convert_to_tensor=True
            )
        return query_embeddings, doc_embeddings
    
    def _get_embeddings(self, query_inputs, document_inputs) -> Tuple[torch.Tensor, torch.Tensor]:
        if ("embedding" in query_inputs) and ("embedding" in document_inputs):
            return query_inputs["embedding"], document_inputs["embedding"]
        if self.args.hn_filter_model == "stella":
            return self._get_embeddings_stella(query_inputs, document_inputs)
        elif self.args.hn_filter_model == "nomic":
            return self._get_embeddings_nomic(query_inputs, document_inputs)
        else:
            return self._get_embeddings_sentence_transformers(query_inputs, document_inputs)

    def _get_query_doc_scores(self, query_inputs, document_inputs) -> torch.Tensor:
        query_embeddings, doc_embeddings = self._get_embeddings(query_inputs, document_inputs)
        if self.args.hn_filter_model == "stella":
            return self._hn_filter_model.similarity(query_embeddings, doc_embeddings)
        else:
            return query_embeddings @ doc_embeddings.T