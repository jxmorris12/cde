from typing import Dict

import collections
import logging
import math

import datasets
import torch
import transformers

from lib.dist import gather, get_rank, get_world_size
from lib.tensor import forward_batched
from lib.misc import tqdm_if_main_worker


class RerankHelper:
    """Wraps our model and does reranking.
    
    Template: https://github.com/beir-cellar/beir/blob/main/beir/reranking/rerank.py#L7
    """
    def __init__(self, model: torch.nn.Module, tokenizer: transformers.PreTrainedTokenizer, batch_size: int, max_seq_length: int, name: str, fake_dataset_info: bool):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.name = name
        self.fake_dataset_info = fake_dataset_info
        # Subsample queries from large sets so that we can evaluate
        # in a reasonable amount of time. Also remember this will be
        # distributed across GPUs. So it's not that bad.
        self.max_reranking_queries = 32 # 256
    
    def _forward_batched(self, **kwargs) -> torch.Tensor:
        return forward_batched(
            model=self.model,
            batch_size=self.batch_size,
            **kwargs,
        )
    
    @torch.no_grad
    def rerank(self, 
               corpus: Dict[str, Dict[str, str]], 
               corpus_embeddings: datasets.Dataset,
               queries: Dict[str, str],
               query_embeddings: datasets.Dataset,
               results: Dict[str, Dict[str, float]],
               top_k: int) -> Dict[str, Dict[str, float]]:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pair_ids = []
        rerank_scores_biencoder = []

        query_idx_dict = {key: j for j, key in enumerate(queries['id'])} # Map IDs to idxs
        corpus_idx_dict = {key: j for j, key in enumerate(corpus['id'])} # Map IDs to idxs
        
        #### Starting to Rerank using cross-attention
        logging.info("Starting To Rerank Top-{}....".format(top_k))

        big_neg_number = -10**10
        
        rank = get_rank()
        world_size = get_world_size()
        query_keys = sorted(list(results.keys()))
        doc_id_to_key = collections.defaultdict(dict)
        num_eval_queries = min(self.max_reranking_queries, len(query_keys))
        for j, query_id in tqdm_if_main_worker(enumerate(query_keys), total=num_eval_queries, desc=f"[{self.name}]"):
            topk_docs = sorted(results[query_id].items(), key=lambda item: item[1], reverse=True)[:top_k]

            # TODO: Enable assertion
            # assert len(topk_docs) == topk_docs, f"fewer than {top_k} docs available in dataset {self.name}"

            for doc_j, (doc_id, _) in enumerate(topk_docs):
                doc_id_to_key[query_id][doc_j] = doc_id

            if j >= self.max_reranking_queries:
                break

            if (j % world_size) != rank:
                # poor man's distributed sampler
                continue
            query_text = queries[query_idx_dict[query_id]]["text"]
            documents_text = []
            for doc_j, (doc_id, _) in enumerate(topk_docs):
                # pair_ids.append([int(query_id), int(doc_id)])
                pair_ids.append([(j, doc_j)])
                doc_j += 1
                documents_text.append(corpus[corpus_idx_dict[doc_id]]["text"])

            query_inputs = self.tokenizer(
                [query_text],
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt",
            ).to(device)
            document_inputs = self.tokenizer(
                documents_text,
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt",
            ).to(device)

            if self.fake_dataset_info:
                batch_size = document_inputs["input_ids"].shape[0]
                fake_seq_length = 128
                fake_dataset_input_ids = torch.ones(
                    (batch_size, fake_seq_length), device=document_inputs["input_ids"].device,
                    dtype=torch.long
                )
                fake_dataset_attention_mask = torch.ones(
                    (batch_size, fake_seq_length), device=document_inputs["input_ids"].device,
                    dtype=torch.long
                )
                dataset_input_ids = fake_dataset_input_ids
                dataset_attention_mask = fake_dataset_attention_mask
            else:
                dataset_input_ids = document_inputs.input_ids
                dataset_attention_mask = document_inputs.attention_mask

            if len(dataset_input_ids) < top_k:
                # TODO: Fix so we always have this and don't need to hack like this
                dataset_input_ids = torch.cat([dataset_input_ids, dataset_input_ids], dim=0)[:top_k]
                dataset_attention_mask = torch.cat([dataset_attention_mask, dataset_attention_mask], dim=0)[:top_k]
            
            # TODO: Cache first-stage outputs here so that things are ~30% faster.
            query_embedding = self._forward_batched(
                input_ids=query_inputs.input_ids,
                attention_mask=query_inputs.attention_mask,
                dataset_input_ids=dataset_input_ids,
                dataset_attention_mask=dataset_attention_mask,
            ).flatten()
            document_embeddings = self._forward_batched(
                input_ids=document_inputs.input_ids,
                attention_mask=document_inputs.attention_mask,
                dataset_input_ids=dataset_input_ids,
                dataset_attention_mask=dataset_attention_mask,
            )
            
            biencoder_score = torch.nn.functional.cosine_similarity(
                query_embedding, 
                document_embeddings
            ).flatten().cpu()
            rerank_scores_biencoder.extend(biencoder_score.tolist())
        
        pair_ids = torch.tensor(pair_ids, device=device)
        rerank_scores_biencoder = torch.tensor(rerank_scores_biencoder, device=device)

        # use actual top-k docs (in case we have fewer)
        # TODO: check this logic
        true_top_k = len(documents_text)
        
        num_queries = min(self.max_reranking_queries, len(queries))
        max_length = int(math.ceil(true_top_k * num_queries / world_size) * 2)

        # add dummy elements to make same shapes for gather.
        extra_ones_pair = (
            torch.ones((max_length - len(pair_ids), *pair_ids.shape[1:]), device=device, dtype=torch.long)
            * big_neg_number
        ) 
        pair_ids = torch.cat(
            (pair_ids, extra_ones_pair),
            dim=0
        )
        extra_ones_rerank = (
            torch.ones(
                (max_length - len(rerank_scores_biencoder), *rerank_scores_biencoder.shape[1:]), 
                device=device, dtype=torch.float32
            ) * big_neg_number
        )
        rerank_scores_biencoder = torch.cat(
            (rerank_scores_biencoder, extra_ones_rerank),
            dim=0
        )
        pair_ids = gather(pair_ids).cpu()
        rerank_scores_biencoder = gather(rerank_scores_biencoder).cpu()

        pair_ids = pair_ids.reshape((-1, 2))
        rerank_scores_biencoder = rerank_scores_biencoder.flatten()

        pair_ids = pair_ids[pair_ids[:, 0] > big_neg_number].tolist()
        rerank_scores_biencoder = rerank_scores_biencoder[rerank_scores_biencoder > big_neg_number].tolist()

        #### Reranking results
        rerank_results_biencoder = collections.defaultdict(dict)
        for pair, score in zip(pair_ids, rerank_scores_biencoder):
            query_j, doc_j = pair[0], pair[1]
            query_id = query_keys[query_j]
            doc_id = doc_id_to_key[query_id][doc_j]
            rerank_results_biencoder[query_id][doc_id] = score
        # TODO: Add assertions here.
        
        # if rank == 0: 
        #     breakpoint()
        # torch.distributed.barrier()
        
        return rerank_results_biencoder