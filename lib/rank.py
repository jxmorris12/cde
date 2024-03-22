from typing import Dict

import collections
import logging
import math

import datasets
import torch
import transformers

from . import gather, get_rank, get_world_size, tqdm_if_main_worker 


class RerankHelper:
    """Wraps our model and does reranking.
    
    Template: https://github.com/beir-cellar/beir/blob/main/beir/reranking/rerank.py#L7
    """
    def __init__(self, model: torch.nn.Module, tokenizer: transformers.PreTrainedTokenizer, batch_size: int, max_seq_length: int, name: str):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.name = name
        self.max_reranking_queries = 2 # 500
    
    def _score(self, query_embedding: torch.Tensor, corpus_embeddings: torch.Tensor) -> float:
        with torch.no_grad():
            scores = self.model(
                query_embedding=query_embedding, 
                document_embeddings=corpus_embeddings
            )
        return scores.flatten().cpu().tolist()
    
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
        for j, query_id in tqdm_if_main_worker(enumerate(query_keys), total=min(self.max_reranking_queries, len(query_keys)), desc=f"[{self.name}]"):
            topk_docs = sorted(results[query_id].items(), key=lambda item: item[1], reverse=True)[:top_k]
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
            
            with torch.no_grad():
                query_embedding = self.model(
                    input_ids=query_inputs.input_ids,
                    attention_mask=query_inputs.attention_mask,
                    dataset_input_ids=document_inputs.input_ids,
                    dataset_attention_mask=document_inputs.attention_mask,
                ).flatten()
                document_embeddings = self.model(
                    input_ids=document_inputs.input_ids,
                    attention_mask=document_inputs.attention_mask,
                    dataset_input_ids=document_inputs.input_ids,
                    dataset_attention_mask=document_inputs.attention_mask,
                )
            
            biencoder_score = torch.nn.functional.cosine_similarity(
                query_embedding, 
                document_embeddings
            ).flatten().cpu().tolist()
            rerank_scores_biencoder.extend(biencoder_score)
        
        pair_ids = torch.tensor(pair_ids, device=device)
        rerank_scores_biencoder = torch.tensor(rerank_scores_biencoder, device=device)
        
        num_queries = min(self.max_reranking_queries, len(queries))
        max_length = int(math.ceil(top_k * num_queries / world_size) * 2)

        # add dummy elements to make same shapes for gather.
        extra_ones_pair = (
            torch.ones((max_length - len(pair_ids), *pair_ids.shape[1:]), device=device, dtype=torch.long)) * big_neg_number
        pair_ids = torch.cat(
            (pair_ids, extra_ones_pair),
            dim=0
        )
        extra_ones_rerank = torch.ones(
            (max_length - len(rerank_scores_biencoder), *rerank_scores_biencoder.shape[1:]), device=device, dtype=torch.float32) * big_neg_number
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
        for pair, biencoder_score in zip(pair_ids, rerank_scores_biencoder):
            query_j, doc_j = pair[0], pair[1]
            query_id = query_keys[query_j]
            doc_id = doc_id_to_key[query_id][doc_j]
            rerank_results_biencoder[query_id][doc_id] = biencoder_score
        
        return rerank_results_biencoder