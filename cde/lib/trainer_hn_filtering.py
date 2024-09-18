from typing import Optional, Tuple

import glob
import os

import datasets
import pyarrow as pa
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer
import torch
import transformers

from cde.lib import (
    gather,
    get_cde_cache_dir, 
    get_rank,
    get_world_size,
    mean_pool,
    md5_hash_kwargs, 
    print0, 
    tqdm_if_main_worker
)

def save_embeddings(embeddings: torch.Tensor, file_path: str):
    table = pa.Table.from_arrays([pa.array(embeddings)], names=['embeddings'])
    # Write to Parquet file
    pq.write_table(table, file_path)
    print(f"Saved {len(embeddings)} embeddings to {file_path}.")


MAX_NUM_EMBEDDINGS_PER_SHARD = 2_000_000

class NomicEmbeddingModelWrapper(torch.nn.Module):
    # TODO: Put this somewhere else
    def __init__(self):
        super().__init__()
        self.model = transformers.AutoModel.from_pretrained(
            "nomic-ai/nomic-embed-text-v1", 
            trust_remote_code=True
        )
        self.model.eval()
    
    def forward(self, **kwargs):
        output = self.model(**kwargs)
        embeddings = mean_pool(output[0], kwargs["attention_mask"])
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)


class TrainerNegativeFilterMixin:
    """Filters hard negatives based on another pre-trained model."""
    _initialized = False
    _hn_filter_model: Optional[torch.nn.Module] = None
    _hn_on_scratch: bool = False
    @property
    def _filename_hash(self) -> str:
        return md5_hash_kwargs(
            dataset_fingerprint=self.train_dataset._fingerprint,
            hn_filter_model=self.args.hn_filter_model,
            seq_length=self.model.config.max_seq_length,
        )

    @property
    def _filename_dir(self) -> str:
        return os.path.join(get_cde_cache_dir(), "hn_embeddings", self._filename_hash)

    @property
    def _scratch_filename_dir(self) -> str:
        return os.path.join(get_cde_cache_dir(), "hn_embeddings", self._filename_hash)
        # return os.path.join("/scratch", "jxm", "cde", "hn_embeddings", self._filename_hash)

    @property
    def _scratch_filename_document(self) -> str:
        return os.path.join(self._scratch_filename_dir, f"corpus_final.parquet")
    
    @property
    def _scratch_filename_query(self) -> str:
        return os.path.join(self._scratch_filename_dir, f"query_final.parquet")

    @property
    def _scratch_cache_dir(self) -> str:
        return os.path.join("/scratch", "jxm", ".cache", "huggingface", "datasets")
    
    def _filename_doc_index(self, shard: int, scratch: bool = False) -> str:
        if scratch:
            return os.path.join(self._scratch_filename_dir, f"corpus_shard_{shard}.parquet")
        else:
            return os.path.join(self._filename_dir, f"corpus_shard_{shard}.parquet")

    def _filename_query_index(self, shard: int, scratch: bool = False) -> str:
        if scratch:
            return os.path.join(self._scratch_filename_dir, f"query_shard_{shard}.parquet")
        else:
            return os.path.join(self._filename_dir, f"query_shard_{shard}.parquet")
    
    @property
    def _filename_finished_sentinel(self) -> str:
        return os.path.join(self._filename_dir, "finished")
    
    def _move_hn_to_scratch(self):
        scratch_dir = self._scratch_filename_dir
        os.makedirs(scratch_dir, exist_ok=True)
        if os.path.exists(self._scratch_filename_document) and os.path.exists(self._scratch_filename_query):
            print0(f"[trainer_hn_filter] Found query files hn on scratch at {scratch_dir}. Continuing.")
            return
        print0(f"[trainer_hn_filter] Moving hn to scratch at {scratch_dir}. This is a one-time operation on this machine.")
        
        datasets.utils.enable_progress_bars()
        
        num_shards = len(glob.glob(os.path.join(self._filename_dir, "corpus_shard_*.parquet")))
        if not os.path.exists(self._scratch_filename_document):
            print0(f"[trainer_hn_filter] Loading document embeddings from {self._filename_dir}...")
            document_embedding_index = datasets.load_dataset(
                "parquet", 
                data_files=[self._filename_doc_index(shard, scratch=False) for shard in range(num_shards)], 
                keep_in_memory=False,
                num_proc=32,
            )["train"]
            document_embedding_index = document_embedding_index.rename_column("embeddings", "document_embedding")
            print0(f"[trainer_hn_filter] Loaded document embeddings from {self._filename_dir}...")
            document_embedding_index = document_embedding_index.flatten_indices()
            # document_embedding_index.set_format(type="torch")
            num_cleaned_files = document_embedding_index.cleanup_cache_files()
            document_embedding_index.save_to_disk(
                self._scratch_filename_document, 
                # max_shard_size="10MB",
                # num_proc=32,
            )
            print0(f"[trainer_hn_filter] Cleaned {num_cleaned_files} files and saved document embeddings to {self._scratch_filename_document}.")

        if not os.path.exists(self._scratch_filename_query):
            print0(f"[trainer_hn_filter] Loading query embeddings from {self._filename_dir}...")
            query_embedding_index = datasets.load_dataset(
                "parquet", 
                data_files=[self._filename_query_index(shard, scratch=False) for shard in range(num_shards)],
                keep_in_memory=False,
                num_proc=32,
            )["train"]
            # TODO: Make this faster by queries.save_to_disk(queries_path) and then loading it back in
            query_embedding_index = query_embedding_index.rename_column("embeddings", "query_embedding")
            print0(f"[trainer_hn_filter] Loaded query embeddings from {self._filename_dir}...")
            query_embedding_index = query_embedding_index.flatten_indices()
            # query_embedding_index.set_format(type="torch")
            num_cleaned_files = query_embedding_index.cleanup_cache_files()
            query_embedding_index.save_to_disk(
                self._scratch_filename_query, 
                # max_shard_size="10MB",
                # num_proc=32,
            )
            print0(f"[trainer_hn_filter] Cleaned {num_cleaned_files} files and saved query embeddings to {self._scratch_filename_query}.")

    def _load_hn_index(self):
        if self._is_main_worker:
            self._move_hn_to_scratch()
        try:
            torch.distributed.barrier()
        except ValueError:
            pass # not in DDP
        
        document_embedding_index = datasets.load_from_disk(
            self._scratch_filename_document, 
            keep_in_memory=False
        )
        query_embedding_index = datasets.load_from_disk(
            self._scratch_filename_query, 
            keep_in_memory=False
        )

        og_fingerprint = self.train_dataset.dataset._fingerprint
        self.train_dataset.dataset = datasets.concatenate_datasets([
            self.train_dataset.dataset, document_embedding_index, query_embedding_index],
            axis=1
        )
        self.train_dataset.dataset.set_format(type="torch")
        # self.train_dataset.dataset.flatten_indices(
        #     cache_file_name=os.path.join(self._scratch_cache_dir, self.train_dataset.dataset._fingerprint), 
        #     writer_batch_size=20_000
        # )
        self.train_dataset.dataset._fingerprint = og_fingerprint
    
    def _init_hn_filter_model(self):
        if self._initialized: return
        if (self._hn_filter_model is None):
            if self.args.hn_filter_model == "nomic_st":
                self._hn_filter_model = SentenceTransformer(
                    "nomic-ai/nomic-embed-text-v1", 
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    trust_remote_code=True,
                )
                self._hn_filter_model.to(self.args.device)
            elif self.args.hn_filter_model == "nomic":
                self._hn_filter_model = NomicEmbeddingModelWrapper()
                self._hn_filter_model.to(self.args.device)
            elif self.args.hn_filter_model == "stella":
                self._hn_filter_model = SentenceTransformer(
                    "dunzhang/stella_en_400M_v5", 
                    trust_remote_code=True, 
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    model_kwargs={
                        "torch_dtype": torch.bfloat16,
                        # "attn_implementation": "flash_attention_2",
                    }
                )
                print0(f"[trainer_hn_filter] Loaded model stella_en_400M_v5 and set max_seq_length to {self.model.config.max_seq_length}.")
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
        
        self._initialized = True

        if not self.args.hn_filter_precompute_vectors:
            return
        
        print0(f"[trainer_hn_filter] Checking for sentinel file {self._filename_finished_sentinel}.")
        if not os.path.exists(self._filename_finished_sentinel):
            print0(f"[trainer_hn_filter] Precomputing hn index for {self.args.hn_filter_model} (max_seq_length={self.model.config.max_seq_length}, batch_size={self._inference_batch_size})...")
            print0(f"[trainer_hn_filter] shard size = {MAX_NUM_EMBEDDINGS_PER_SHARD}.")
            self._precompute_hn_index()
        
        self._load_hn_index()
        print0(f"[trainer_hn_filter] Loaded embedding index. Num cache files = {len(self.train_dataset.dataset.cache_files)}")
    
    @property
    def _inference_batch_size(self) -> int:
        gc_bs_fs = (
            self.args.max_batch_size_fits_in_memory_first_stage 
            or 
            self.args.max_batch_size_fits_in_memory
        )
        return min(
            gc_bs_fs, self.args.per_device_train_batch_size)
        
    def _precompute_hn_index(self):
        batch_size = self._inference_batch_size
        # Precompute vectors for hard negatives
        train_dataset_length = (len(self.train_dataset) // (batch_size * get_world_size())) * batch_size * get_world_size()
        print0(f"Precomputing embeddings for {len(self.train_dataset)} samples: {train_dataset_length} across {get_world_size()} workers and {len(self.train_dataset) - train_dataset_length} on self.")
        dataset1 = self.train_dataset.dataset.select(range(0, train_dataset_length))
        train_dataloader_1 = torch.utils.data.DataLoader(
            dataset1, 
            batch_size=4096, 
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
                
                # gc.collect()
                # torch.cuda.empty_cache()
                print0(f"[trainer_hn_filter] Saved {len(all_query_embeddings)} query+doc embeddings to disk (shard={shard_idx}).")
                shard_idx += 1
                all_query_embeddings, all_doc_embeddings = [], []
        # Compute last samples on main worker to even it out
        dataset2 = self.train_dataset.dataset.select(range(train_dataset_length, len(self.train_dataset)))
        train_dataloader_2 = torch.utils.data.DataLoader(
            dataset2, 
            batch_size=4096, 
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
        self._init_hn_filter_model()
        model = self._hn_filter_model
        queries = query_inputs["text"]
        docs = document_inputs["text"]
        with torch.no_grad():
            query_embeddings = model.encode(
                queries,
                batch_size=self._inference_batch_size,
                convert_to_tensor=True
            )
            doc_embeddings = model.encode(
                docs,    
                batch_size=self._inference_batch_size,
                convert_to_tensor=True
            )
        return query_embeddings, doc_embeddings
            
    def _get_embeddings_nomic(self, query_inputs, document_inputs) -> Tuple[torch.Tensor, torch.Tensor]:
        from cde.lib.tensor import forward_batched
        
        if self._hn_filter_model is None:
            self._hn_filter_model = NomicEmbeddingModelWrapper()
            self._hn_filter_model.to(self.args.device)

        with torch.no_grad():
            query_outputs = forward_batched(
                model=self._hn_filter_model,
                input_ids=query_inputs["input_ids"][:, :512],
                attention_mask=query_inputs["attention_mask"][:, :512],
                batch_size=(self._inference_batch_size * 2),
            )
            doc_outputs = forward_batched(
                model=self._hn_filter_model,
                input_ids=document_inputs["input_ids"][:, :512],
                attention_mask=document_inputs["attention_mask"][:, :512],
                batch_size=(self._inference_batch_size * 2),
            )

        query_embeddings = torch.nn.functional.normalize(query_outputs, p=2, dim=1)
        doc_embeddings = torch.nn.functional.normalize(doc_outputs, p=2, dim=1)
        return query_embeddings, doc_embeddings

    def _get_embeddings_nomic_st(self, query_inputs, document_inputs) -> Tuple[torch.Tensor, torch.Tensor]:
        self._init_hn_filter_model()
    
        queries = query_inputs["text"]
        docs = document_inputs["text"]
        
        model = self._hn_filter_model
        with torch.no_grad():
            query_embeddings = model.encode(
                [f"search_query: {q}" for q in queries],    
                batch_size=(self._inference_batch_size),
                convert_to_tensor=True
            )
            doc_embeddings = model.encode(
                [f"search_document: {d}" for d in docs],    
                batch_size=(self._inference_batch_size),
                convert_to_tensor=True
            )

        query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
        doc_embeddings = torch.nn.functional.normalize(doc_embeddings, p=2, dim=1)
        return query_embeddings, doc_embeddings
    
    def _get_embeddings_stella(self, query_inputs, document_inputs) -> Tuple[torch.Tensor, torch.Tensor]:
        # https://huggingface.co/dunzhang/stella_en_400M_v5
        self._init_hn_filter_model()
    
        queries = query_inputs["text__no_prefix"]
        docs = document_inputs["text__no_prefix"]

        query_prefix = query_inputs["prefix"][0]
        document_prefix = document_inputs["prefix"][0]
        
        if query_prefix == document_prefix:
            query_prompt_name = "s2s_query"
        else:
            query_prompt_name = "s2p_query"

        model = self._hn_filter_model
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            query_embeddings = model.encode(
                queries,
                prompt_name=query_prompt_name, 
                batch_size=self._inference_batch_size,
                convert_to_tensor=True
            )
            doc_embeddings = model.encode(
                docs,    
                batch_size=self._inference_batch_size,
                convert_to_tensor=True
            )
        return query_embeddings, doc_embeddings
    
    def _get_embeddings(self, query_inputs, document_inputs) -> Tuple[torch.Tensor, torch.Tensor]:
        # print("query_inputs.keys() =>", query_inputs.keys())
        # print("document_inputs.keys() =>", document_inputs.keys())
        # print()
        if ("embedding" in query_inputs) and ("embedding" in document_inputs):
            return query_inputs["embedding"], document_inputs["embedding"]
        if self.args.hn_filter_model == "stella":
            return self._get_embeddings_stella(query_inputs, document_inputs)
        elif self.args.hn_filter_model == "nomic":
            return self._get_embeddings_nomic(query_inputs, document_inputs)
        elif self.args.hn_filter_model == "nomic_st":
            return self._get_embeddings_nomic_st(query_inputs, document_inputs)
        else:
            return self._get_embeddings_sentence_transformers(query_inputs, document_inputs)

    def _get_query_doc_scores(self, query_inputs, document_inputs) -> torch.Tensor:
        query_embeddings, doc_embeddings = self._get_embeddings(query_inputs, document_inputs)
        query_embeddings = query_embeddings.to(self.args.device)
        doc_embeddings = doc_embeddings.to(self.args.device)
        if self.args.hn_filter_model == "stella":
            return self._hn_filter_model.similarity(query_embeddings, doc_embeddings)
        else:
            return query_embeddings @ doc_embeddings.T
