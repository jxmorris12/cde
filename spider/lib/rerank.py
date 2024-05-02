from typing import Dict, List

import collections
import functools
import heapq
import logging
import math

import torch
import tqdm
import transformers

from spider.lib.dist import gather, get_rank, get_world_size
from spider.lib.embed import embed_with_cache
from spider.lib.tensor import forward_batched
from spider.lib.misc import tqdm_if_main_worker


def tokenize_transform(
        examples: Dict[str, List],
        tokenizer,
        prefix: str,
        col: str,
        max_length: int,
    ) -> Dict[str, torch.Tensor]:
    texts = examples[col]
    batch_dict = tokenizer(
        [prefix + t for t in texts],
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    return batch_dict
    

class RerankHelper:
    """Wraps our model and does reranking.
    
    Template: https://github.com/beir-cellar/beir/blob/main/beir/reranking/rerank.py#L7
    """
    def __init__(self, 
            model: torch.nn.Module, 
            tokenizer: transformers.PreTrainedTokenizer, 
            batch_size: int, 
            max_reranking_queries: int, 
            max_seq_length: int, 
            name: str, 
            fake_dataset_info: bool
        ):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.name = name
        self.fake_dataset_info = fake_dataset_info
        # Subsample queries from large sets so that we can evaluate
        # in a reasonable amount of time. Also remember this will be
        # distributed across GPUs. So it's not that bad.
        self.max_reranking_queries = max_reranking_queries
    
    def _forward_batched(self, **kwargs) -> torch.Tensor:
        return forward_batched(
            model=self.model,
            batch_size=self.batch_size,
            **kwargs,
        )


    @torch.no_grad
    def rerank(self, dataset, top_k: int) -> Dict[str, Dict[str, float]]:
        tokenize_corpus_func = functools.partial(
            tokenize_transform, 
            col="text",
            prefix=dataset.prefix_document,
            tokenizer=self.tokenizer,
            max_length=self.max_seq_length,
        )
        corpus: datasets.Dataset = dataset.corpus.map(tokenize_corpus_func, batched=True)
        corpus.set_format("pt")

        tokenize_queries_func = functools.partial(
            tokenize_transform, 
            col="text",
            prefix=dataset.prefix_query,
            tokenizer=self.tokenizer,
            max_length=self.max_seq_length,
        )
        queries: datasets.Dataset = dataset.queries.map(tokenize_queries_func, batched=True)
        queries.set_format("pt")

        results: Dict[str, Dict[str, float]] = dataset.rerank_results
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
        agreement = []
        for j, query_id in tqdm_if_main_worker(enumerate(query_keys), total=num_eval_queries, desc=f"[{self.name}]"):
            topk_docs = sorted(results[query_id].items(), key=lambda item: item[1], reverse=True)[:top_k]

            # TODO: Enable assertion...
            # assert len(topk_docs) == topk_docs, f"fewer than {top_k} docs available in dataset {self.name}"

            for doc_j, (doc_id, _) in enumerate(topk_docs):
                doc_id_to_key[query_id][doc_j] = doc_id

            if j >= self.max_reranking_queries:
                break

            if (j % world_size) != rank:
                # poor man's distributed sampler
                continue

            query_batch = queries[query_idx_dict[query_id]]
            query_inputs = transformers.BatchEncoding(
                data={
                    "input_ids": query_batch["input_ids"],
                    "attention_mask": query_batch["attention_mask"],
                }
            ).to(device)

            documents_input_ids = []
            documents_attention_mask = []

            pad = lambda t: torch.nn.functional.pad(t, pad=[0, self.max_seq_length - t.shape[-1]], value=self.tokenizer.pad_token_id)
            for doc_j, (doc_id, _) in enumerate(topk_docs):
                pair_ids.append([(j, doc_j)])
                doc_j += 1
                documents_input_ids.append(pad(corpus[corpus_idx_dict[doc_id]]["input_ids"]))
                documents_attention_mask.append(pad(corpus[corpus_idx_dict[doc_id]]["attention_mask"]))
            document_inputs = transformers.BatchEncoding(data={
                "input_ids": torch.stack(documents_input_ids),
                "attention_mask": torch.stack(documents_attention_mask)
            }).to(device)

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
                input_ids=query_inputs.input_ids[None],
                attention_mask=query_inputs.attention_mask[None],
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

            agreement.append(biencoder_score.argmax() == 0)
        
        # print("agreement perc:", torch.tensor(agreement).float().mean())
        pair_ids = torch.tensor(pair_ids, device=device)
        rerank_scores_biencoder = torch.tensor(rerank_scores_biencoder, device=device)

        true_top_k = len(document_inputs.input_ids)
        max_length = int(math.ceil(true_top_k * num_eval_queries / world_size)) * 2

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
        return rerank_results_biencoder
    

def get_reranking_results(data_path: str, split: str, model_name: str) -> Dict:
    """Reranks dataset at `data_path` using model with name `model_name`."""
    # Reranking adapted from here.
    # https://github.com/embeddings-benchmark/mteb/blob/0c67d969b8e34ccf94286c3e2758c7a8d0943e81/mteb/evaluation/evaluators/RetrievalEvaluator.py#L189
    from mteb import HFDataLoader, RetrievalEvaluator

    print("[get_reranking_results Loading:", data_path)
    corpus, queries, _qrels = HFDataLoader(data_folder=data_path, streaming=False, keep_in_memory=False).load(split=split)
    retriever = RetrievalEvaluator(
        None,
        score_function="dot",
        batch_size=2048,
        # This way we'll keep the top 10,000 not just
        #  the top 1,000 (by default) values.
        k_values=[1, 3, 5, 10, 100, 1000, 10_000],
        corpus_chunk_size=2**20,
    )

    query_embeddings = embed_with_cache(
        model_name=model_name, 
        cache_name='????pathdoesntexist', # TODO cleaner way to specify.
        d=queries, 
        col='text',
        save_to_disk=False,
        batch_size=4096,
    )["embeds"]
    
    # queries = { query['id']: query['text'] for query in queries }
    # corpus = { doc['id']: {'title': doc['title'] , 'text': doc['text']} for doc in corpus }
    
    # Get top-k values
    query_ids = list(queries["id"])
    corpus_ids = corpus["id"]
    results = {qid: {} for qid in query_ids}
    corpus_chunk_size = 1_000_000
    top_k = retriever.top_k
    result_heaps = {
        qid: [] for qid in query_ids
    }  # Keep only the top-k docs for each query

    for corpus_start_idx in tqdm.trange(0, len(corpus), corpus_chunk_size, colour="#BF40BF", desc=f"evaluating {data_path}"):
        corpus_end_idx = min(corpus_start_idx + corpus_chunk_size, len(corpus))
        sub_corpus_embeddings = embed_with_cache(
            model_name=model_name, 
            cache_name='????pathdoesntexist', # TODO cleaner way to ignore cache.
            d=corpus.select(range(corpus_start_idx, corpus_end_idx)), 
            col='text',
            save_to_disk=False,
            batch_size=4096,
        )["embeds"]
        cos_scores = retriever.retriever.score_functions["cos_sim"](
            query_embeddings, sub_corpus_embeddings
        )
        cos_scores[torch.isnan(cos_scores)] = -1
        cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(
            cos_scores,
            min(top_k + 1, len(cos_scores[1])),
            dim=1,
            largest=True,
            sorted=False,
        )
        cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
        cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()

        for query_itr in range(len(query_embeddings)):
            query_id = query_ids[query_itr]
            for sub_corpus_id, score in zip(
                cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]
            ):
                try:
                    corpus_id = corpus_ids[corpus_start_idx + sub_corpus_id]
                except IndexError:
                    breakpoint()
                if corpus_id != query_id:
                    if len(result_heaps[query_id]) < top_k:
                        # Push item on the heap
                        heapq.heappush(result_heaps[query_id], (score, corpus_id))
                    else:
                        # If item is larger than the smallest in the heap, push it on the heap then pop the smallest element
                        heapq.heappushpop(
                            result_heaps[query_id], (score, corpus_id)
                        )

    for qid in result_heaps:
        for score, corpus_id in result_heaps[qid]:
            results[qid][corpus_id] = score

    return results
