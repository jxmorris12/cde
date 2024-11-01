from typing import Any, Dict, List, Union, Tuple

import collections
import functools
import heapq
import logging
import math
import os

import datasets
import random
import torch
import tqdm
import transformers

from cde.lib.dist import gather, get_rank, get_world_size
from cde.lib.embed import embed_with_cache
from cde.lib.misc import tqdm_if_main_worker
from cde.lib.tensor import forward_batched

from mteb import HFDataLoader, RetrievalEvaluator

MtebDataset = Any # Weird import error so we fake the class name for typing.
BeirDataset = Any # Weird import error so we fake the class name for typing.


TensorPair = Tuple[torch.Tensor, torch.Tensor]


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
            contextual_input_strategy: str,
            pooling_factor: int = 1,
            contextual_n_outputs_ensemble: int = 1,
            default_dtype = torch.bfloat16,
        ):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.name = name

        assert contextual_input_strategy in ["fake", "dummy", "random_corpus", "random_corpus__topk__interp", "topk", "topk_pool", "null", "null_topk"]
        self.contextual_input_strategy = contextual_input_strategy
        # Subsample queries from large sets so that we can evaluate
        # in a reasonable amount of time. Also remember this will be
        # distributed across GPUs. So it's not that bad.
        self.max_reranking_queries = max_reranking_queries
        self.pooling_factor = 4
        self.contextual_n_outputs_ensemble = contextual_n_outputs_ensemble
        self._pad = lambda t: torch.nn.functional.pad(t, pad=[0, self.max_seq_length - t.shape[-1]], value=self.tokenizer.pad_token_id)
        # self.default_dtype = torch.float16
        self.default_dtype = torch.bfloat16
        self.dummy_model = None
    
    def _forward_batched(
            self, 
            model: torch.nn.Module,
            **kwargs
        ) -> torch.Tensor:
        rerank_device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        with torch.autocast(rerank_device, dtype=self.default_dtype):
            return forward_batched(
                model=model,
                batch_size=self.batch_size,
                **kwargs
            )

    def _get_dataset_inputs(
            self, 
            corpus_idx_dict: Dict[str, int], 
            corpus: datasets.Dataset, 
            all_topk_docs: List[Tuple[int, torch.Tensor]], 
            document_inputs: Dict[str, torch.Tensor], 
            tokenize_corpus_func,
            top_k: int
        ) -> TensorPair:
        if self.contextual_input_strategy == "fake":
            batch_size = document_inputs["input_ids"].shape[0]
            fake_seq_length = top_k
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
        elif self.contextual_input_strategy == "random_corpus":
            corpus_ids = random.choices(range(len(corpus)), k=top_k)
            corpus_inputs = tokenize_corpus_func(corpus[corpus_ids])
            corpus_inputs = corpus_inputs.to(document_inputs["input_ids"].device)
            dataset_input_ids = corpus_inputs.input_ids
            dataset_attention_mask = corpus_inputs.attention_mask
        elif self.contextual_input_strategy == "random_corpus__topk__interp":
            corpus_ids = random.choices(range(len(corpus)), k=top_k)
            corpus_inputs = tokenize_corpus_func(corpus[corpus_ids])
            corpus_inputs = corpus_inputs.to(document_inputs["input_ids"].device)
            random_dataset_input_ids = corpus_inputs.input_ids
            random_dataset_attention_mask = corpus_inputs.attention_mask
            dataset_input_ids = torch.cat(
                [
                    document_inputs.input_ids,
                    random_dataset_input_ids,
                ],
                dim=0
            )
            dataset_attention_mask = torch.cat(
                [
                    document_inputs.attention_mask,
                    random_dataset_attention_mask,
                ],
                dim=0
            )
            idxs = torch.randperm(len(dataset_input_ids))[:top_k]
            dataset_input_ids = dataset_input_ids[idxs]
            dataset_attention_mask = dataset_attention_mask[idxs]
        elif self.contextual_input_strategy == "topk_pool":
            num_docs_to_pool = self.pooling_factor * top_k
            topk_docs = all_topk_docs[:num_docs_to_pool]
            dataset_input_ids = []
            dataset_attention_mask = []
            for _, (doc_id, _) in enumerate(topk_docs):
                dataset_input_ids.append(self._pad(corpus[corpus_idx_dict[doc_id]]["input_ids"]))
                dataset_attention_mask.append(self._pad(corpus[corpus_idx_dict[doc_id]]["attention_mask"]))
            dataset_input_ids = torch.stack(dataset_input_ids).to(document_inputs["input_ids"].device)
            dataset_attention_mask = torch.stack(dataset_attention_mask).to(document_inputs["input_ids"].device)
            dataset_input_ids = dataset_input_ids.reshape(
                (self.pooling_factor, top_k, self.max_seq_length)
            )
            dataset_attention_mask = dataset_attention_mask.reshape(
                (self.pooling_factor, top_k, self.max_seq_length)
            )
        else:
            dataset_input_ids = document_inputs.input_ids
            dataset_attention_mask = document_inputs.attention_mask

        if len(dataset_input_ids) < top_k:
            # TODO: Fix so we always have this and don't need to hack like this
            dataset_input_ids = torch.cat([dataset_input_ids, dataset_input_ids], dim=0)[:top_k]
            dataset_attention_mask = torch.cat([dataset_attention_mask, dataset_attention_mask], dim=0)[:top_k]
        
        # print(self.tokenizer.decode(dataset_input_ids[0], skip_special_tokens=True))
        return dataset_input_ids, dataset_attention_mask

    @torch.no_grad
    def rerank(self, dataset: Union[MtebDataset, BeirDataset], top_k: int) -> Dict[str, Dict[str, float]]:
        tokenize_corpus_func = functools.partial(
            tokenize_transform, 
            col="text",
            prefix=dataset.prefix_document,
            tokenizer=self.tokenizer,
            max_length=self.max_seq_length,
        )

        corpus = dataset.corpus
        # corpus: datasets.Dataset = dataset.corpus.map(
        #     tokenize_corpus_func, 
        #     batched=True, 
        #     batch_size=10_000,
        #     num_proc=(64 if len(dataset.corpus) > 700_000 else None)
        # )
        # corpus: datasets.Dataset = dataset.corpus.map(
        #     tokenize_corpus_func, 
        #     batched=True, 
        #     batch_size=10_000,
        # )
        corpus.set_format("pt")

        tokenize_queries_func = functools.partial(
            tokenize_transform, 
            col="text",
            prefix=dataset.prefix_query,
            tokenizer=self.tokenizer,
            max_length=self.max_seq_length,
        )
        queries: datasets.Dataset = dataset.queries.map(
            tokenize_queries_func, 
            batched=True,
            keep_in_memory=True,
        )
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

        self.model.eval()

        docs = [] # Need to initialize empty list to avoid error
        for j, query_id in tqdm_if_main_worker(
            enumerate(query_keys), total=num_eval_queries, desc=f"[{self.name}]"):
            all_topk_docs = sorted(results[query_id].items(), key=lambda item: item[1], reverse=True)
            for doc_j, (doc_id, _) in enumerate(all_topk_docs):
                doc_id_to_key[query_id][doc_j] = doc_id

            if j >= self.max_reranking_queries:
                break

            if (j % world_size) != rank:
                # poor man's distributed sampler
                continue

            ensemble_query_embedding = []
            ensemble_document_embeddings = []
            for k in range(self.contextual_n_outputs_ensemble):
                topk_docs = all_topk_docs[k*top_k:(k+1)*top_k]

                # TODO: Enable assertion...
                # assert len(topk_docs) == topk_docs, f"fewer than {top_k} docs available in dataset {self.name}"

                query_batch = queries[query_idx_dict[query_id]]["text"]
                query_inputs: transformers.BatchEncoding = tokenize_queries_func({ "text": [query_batch] }).to(device)

                docs = []
                for doc_j, (doc_id, _) in enumerate(topk_docs):
                    pair_ids.append([(j, doc_j)])
                    doc_j += 1
                    docs.append(corpus[corpus_idx_dict[doc_id]]["text"])
                
                if self.contextual_input_strategy == "dummy":
                    from sentence_transformers import SentenceTransformer
                    if self.dummy_model is None:
                        self.dummy_model = SentenceTransformer("Snowflake/snowflake-arctic-embed-l")
                    query_embedding = self.dummy_model.encode(
                        [query_batch], 
                        prompt_name="query", 
                        convert_to_tensor=True
                    )
                    document_embeddings = self.dummy_model.encode(
                        docs,
                        batch_size=512,
                        convert_to_tensor=True
                    )
                    ensemble_query_embedding.append(query_embedding)
                    ensemble_document_embeddings.append(document_embeddings)
                    continue
                
                document_inputs = tokenize_corpus_func({ "text": docs })
                document_inputs: transformers.BatchEncoding = document_inputs.to(device)
                dataset_input_ids, dataset_attention_mask = (
                    self._get_dataset_inputs(
                        corpus_idx_dict=corpus_idx_dict,
                        corpus=corpus,
                        all_topk_docs=all_topk_docs,
                        document_inputs=document_inputs,
                        top_k=top_k,
                        tokenize_corpus_func=tokenize_corpus_func,
                    )
                )
                if hasattr(self.model, 'first_stage_model'):
                    assert top_k >= self.model.second_stage_model.num_corpus_tokens
                    dataset_embeddings = self._forward_batched(
                        model=self.model.first_stage_model,
                        input_ids=dataset_input_ids,
                        attention_mask=dataset_attention_mask,
                    )
                    null_dataset_embedding_query = self.contextual_input_strategy in ["null", "null_topk"]
                    query_embedding = self._forward_batched(
                        model=self.model.second_stage_model,
                        input_ids=query_inputs.input_ids,
                        attention_mask=query_inputs.attention_mask,
                        dataset_embeddings=dataset_embeddings,
                        null_dataset_embedding=null_dataset_embedding_query,
                    ).flatten()
                    null_dataset_embedding_document = self.contextual_input_strategy in ["null"]
                    document_embeddings = self._forward_batched(
                        model=self.model.second_stage_model,
                        input_ids=document_inputs.input_ids,
                        attention_mask=document_inputs.attention_mask,
                        dataset_embeddings=dataset_embeddings,
                        null_dataset_embedding=null_dataset_embedding_document,
                    )
                else:
                    # biencoder
                    query_embedding = self._forward_batched(
                        model=self.model,
                        input_ids=query_inputs.input_ids,
                        attention_mask=query_inputs.attention_mask,
                    ).flatten()
                    document_embeddings = self._forward_batched(
                        model=self.model,
                        input_ids=document_inputs.input_ids,
                        attention_mask=document_inputs.attention_mask,
                    )
                ensemble_query_embedding.append(query_embedding)
                ensemble_document_embeddings.append(document_embeddings)
            
            ensemble_query_embedding = torch.stack(ensemble_query_embedding, dim=0).mean(0)
            ensemble_document_embeddings = torch.stack(ensemble_document_embeddings,  dim=0).mean(0)
            
            biencoder_score = torch.nn.functional.cosine_similarity(
                ensemble_query_embedding, 
                ensemble_document_embeddings
            ).flatten().cpu()
            rerank_scores_biencoder.extend(biencoder_score.tolist())
            agreement.append(biencoder_score.argmax() == 0)
        
        # print("agreement perc:", torch.tensor(agreement).float().mean())
        pair_ids = torch.tensor(pair_ids, device=device)
        rerank_scores_biencoder = torch.tensor(rerank_scores_biencoder, device=device)
        max_length = int(math.ceil(top_k * num_eval_queries / world_size)) * 2
        # true_top_k = len(document_inputs.input_ids)
        # true_top_k = len(docs)
        # max_length = int(math.ceil(true_top_k * num_eval_queries / world_size)) * 2

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
        rerank_tqdm = tqdm_if_main_worker(rerank_scores_biencoder, desc="computing score", leave=False)
        for pair, score in zip(pair_ids, rerank_tqdm):
            query_j, doc_j = pair[0], pair[1]
            query_id = query_keys[query_j]
            doc_id = doc_id_to_key[query_id][doc_j]
            rerank_results_biencoder[query_id][doc_id] = score
        # TODO: Add assertions here.
        return rerank_results_biencoder
    

def get_reranking_results(
        data_path: str, 
        split: str, 
        model_name: str,
        hf_data_loader = None
    ) -> Dict:
    """Reranks dataset at `data_path` using model with name `model_name`."""
    # Reranking adapted from here.
    # https://github.com/embeddings-benchmark/mteb/blob/0c67d969b8e34ccf94286c3e2758c7a8d0943e81/mteb/evaluation/evaluators/RetrievalEvaluator.py#L189

    print("[get_reranking_results] Loading:", data_path)
    if hf_data_loader is None:
        hf_data_loader = HFDataLoader(
            data_folder=data_path, 
            streaming=False, 
            keep_in_memory=False
        )
    corpus, queries, _qrels = hf_data_loader.load(split=split)
    retriever = RetrievalEvaluator(
        None,
        score_function="dot",
        batch_size=512,
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
        batch_size=512,
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
            batch_size=512,
        )["embeds"]
        cos_scores = retriever.retriever.score_functions["cos_sim"](
            query_embeddings.cuda(), sub_corpus_embeddings.cuda()
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
                corpus_id = corpus_ids[corpus_start_idx + sub_corpus_id]
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
