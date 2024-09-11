from __future__ import annotations
from typing import Any, Dict, List, Tuple

import ctypes
import gc
import heapq
import json
import logging
import os
import random
from collections import defaultdict


import datasets
import numpy as np
import pytrec_eval
import torch
import transformers
import tqdm
from sentence_transformers import CrossEncoder

from .Evaluator import Evaluator
from .utils import cos_sim, dot_score, download, hole, mrr, recall_cap, top_k_accuracy

from cde.lib.embed import embed_dataloader

logger = logging.getLogger(__name__)


class TransformExamplesDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: datasets.Dataset, corpus_id_to_idx: dict, tokenizer: transformers.PreTrainedTokenizer, prefix: str, max_num_chars: int, max_length: int, all_nn_ids: np.ndarray, all_doc_embeddings: np.ndarray):
        self.dataset = dataset
        self.corpus_id_to_idx = corpus_id_to_idx
        self.tokenizer = tokenizer
        self.prefix = prefix
        self.max_num_chars = max_num_chars
        self.max_length = max_length
        self.all_nn_ids = all_nn_ids
        self.all_doc_embeddings = all_doc_embeddings
    
    def _transform(self, ex_batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        texts = []
        nn_doc_idxs = []
        titles = ex_batch.get("title", ["" for _ in range(len(ex_batch["text"]))])
        for c_id, title, text in zip(ex_batch["id"], titles, ex_batch["text"]):
            text = "{} {}".format(title, text).strip()
            nn_doc_idxs.append(self.all_nn_ids[self.corpus_id_to_idx[c_id]])
            texts.append(text)
        
        os.environ["TOKENIZERS_PARALLELISM"] = "0"
        
        nn_doc_idxs = np.stack(nn_doc_idxs)
        dataset_embeddings = torch.tensor(self.all_doc_embeddings[nn_doc_idxs], dtype=torch.float32)
        batch_dict = self.tokenizer(
            [self.prefix + t[:self.max_num_chars] for t in texts],
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return  {**batch_dict, "dataset_embeddings": dataset_embeddings}
    
    def __getitems__(self, idx_batch):
        ex_batch = self.dataset[idx_batch]
        return self._transform(ex_batch)
    
    def __len__(self):
        return len(self.dataset)

# Adapted from https://github.com/beir-cellar/beir/blob/f062f038c4bfd19a8ca942a9910b1e0d218759d4/beir/retrieval/search/dense/exact_search.py#L12
class DenseRetrievalExactSearch:
    def __init__(
        self,
        model,
        batch_size: int = 128,
        corpus_chunk_size: int = 50000,
        previous_results: str = None,
        first_stage_model=None,
        **kwargs,
    ):
        # Model is class that provides encode_corpus() and encode_queries()
        self.model = model
        self.first_stage_model = first_stage_model
        self.batch_size = batch_size
        self.score_functions = {"cos_sim": cos_sim, "dot": dot_score}
        self.score_function_desc = {
            "cos_sim": "Cosine Similarity",
            "dot": "Dot Product",
        }
        self.corpus_chunk_size = corpus_chunk_size
        self.previous_results = previous_results
        self.show_progress_bar = kwargs.get("show_progress_bar", True)
        self.convert_to_tensor = kwargs.get("convert_to_tensor", True)
        self.save_corpus_embeddings = kwargs.get("save_corpus_embeddings", False)
        self.corpus_embeddings = defaultdict(list)
        self.results = {}

        if self.previous_results is not None:
            self.previous_results = self.load_results_file()

        if isinstance(self.model, CrossEncoder):
            # load the predict instance from the CrossEncoder
            # custom functions can be used by extending the DenseRetrievalExactSearch class
            self.predict = self.model.predict

    def search(
        self,
        corpus: dict[str, dict[str, str]],
        corpus_cluster_ids: dict[int, List[str]],
        queries: dict[str, str],
        query_cluster_ids: dict[str, int],
        top_k: int,
        score_function: str,
        return_sorted: bool = False,
        **kwargs,
    ) -> dict[str, dict[str, float]]:
        # Create embeddings for all queries using model.encode_queries()
        # Runs semantic search against the corpus embeddings
        # Returns a ranked list with the corpus ids
        if score_function not in self.score_functions:
            raise ValueError(
                "score function: {} must be either (cos_sim) for cosine similarity or (dot) for dot product".format(
                    score_function
                )
            )

        logger.info("Encoding Queries...")
        query_ids = list(queries.keys())
        self.results = {qid: {} for qid in query_ids}

        # Do clustering here.
        all_doc_ids = list(corpus.keys())
        all_docs = [corpus[doc_id] for doc_id in all_doc_ids]
        all_doc_embeddings = self.first_stage_model.encode_corpus(
            all_docs,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_tensor=True,
            output_device="cpu",
        )
        all_doc_embeddings = all_doc_embeddings.numpy()
        # all_doc_embeddings = all_doc_embeddings.astype(np.float16)

        gc.collect()
        
        all_docs_idx = { doc_id: i for i, doc_id in enumerate(all_doc_ids) }
        if hasattr(self.model.encoder, "module"):
            transductive_corpus_size = self.model.encoder.module.config.transductive_corpus_size
        else:
            transductive_corpus_size = self.model.encoder.config.transductive_corpus_size

        # Get first-stage embeddings for 
        # all relevant documents
        query_dataset_embeddings = []
        random_nn_ids = random.sample(list(all_doc_ids), transductive_corpus_size)
        for q_id in tqdm.tqdm(queries.keys(), desc="Collecting nearest neighbors for queries"):
            nn_ids = corpus_cluster_ids[query_cluster_ids[q_id]]
            # nn_ids = random_nn_ids
            if len(nn_ids) < transductive_corpus_size:
                # n_additional_docs = transductive_corpus_size - len(nn_ids)
                nn_ids = nn_ids + random_nn_ids
            nn_ids = nn_ids[:transductive_corpus_size]
            nn_doc_idxs = torch.tensor([all_docs_idx[nn_id] for nn_id in nn_ids])
            query_dataset_embeddings.append(
                torch.tensor(all_doc_embeddings[nn_doc_idxs])
            )
        assert len(set([e.shape for e in query_dataset_embeddings])) == 1

        query_text = [queries[qid] for qid in query_ids]
        query_text_ds = datasets.Dataset.from_dict({ 
                "text": query_text, 
                "dataset_embeddings": query_dataset_embeddings 
            }
        )
        query_text_ds.set_format("torch")
        query_embeddings = self.model.encode_queries(
            query_text_ds,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_tensor=self.convert_to_tensor,
        )

        # logger.info("Sorting Corpus by document length (Longest first)...")
        corpus_ids = list(corpus.keys())
        corpus_text = [corpus[cid] for cid in corpus_ids]

        logger.info("Encoding Corpus in batches... Warning: This might take a while!")
        logger.info(
            "Scoring Function: {} ({})".format(
                self.score_function_desc[score_function], score_function
            )
        )

        # Reverse corpus mapping
        doc_clusters = {}
        for cluster_id, doc_ids in corpus_cluster_ids.items():
            for doc_id in doc_ids:
                doc_clusters[doc_id] = cluster_id
        
        all_nn_ids = np.zeros((len(all_doc_ids), transductive_corpus_size), dtype=np.int64)
        for j, doc_id in tqdm.tqdm(
                enumerate(all_doc_ids), 
                total=len(all_doc_ids), 
                desc="Getting nearest neighbors for all docs"
            ):
            cluster_id = doc_clusters[doc_id]
            nn_ids = corpus_cluster_ids[cluster_id] 
            # nn_ids = random_nn_ids
            if len(nn_ids) < transductive_corpus_size:
                nn_ids = nn_ids + random_nn_ids
            # random.shuffle(nn_ids)
            nn_ids = nn_ids[:transductive_corpus_size]
            all_nn_ids[j] = np.array([all_docs_idx[nn_id] for nn_id in nn_ids])
        
        gc.collect()

        print("[DenseEncoder] Creating dataloader and preparing to encode docs...")
        dataset = datasets.Dataset.from_list([ { "id": _id, **ex } for _id, ex in corpus.items()])

        dataset_wrapped = TransformExamplesDataset(
            dataset=dataset,
            corpus_id_to_idx=dict(zip(all_doc_ids, range(len(all_doc_ids)))),
            tokenizer=self.model.tokenizer, 
            prefix=self.model.document_prefix, 
            max_num_chars=self.model._max_num_chars, 
            max_length=self.model.max_length,
            all_nn_ids=all_nn_ids,
            all_doc_embeddings=all_doc_embeddings,
        )

        num_workers = min(len(os.sched_getaffinity(0)), self.model.gpu_count * 2)
        effective_batch_size = min(self.batch_size // 2 * self.model.gpu_count, max(1, len(dataset) // max(1, num_workers)))
        print(f"[DenseRetrievalExactSearch] making dataloader with num_workers={num_workers} // effective_batch_size = {effective_batch_size}")
        data_collator = transformers.DataCollatorWithPadding(self.model.tokenizer, pad_to_multiple_of=8)
        data_loader = torch.utils.data.DataLoader(
            dataset_wrapped,
            batch_size=effective_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers, 
            prefetch_factor=(4 if num_workers > 0 else None),
            pin_memory=True,
            collate_fn=data_collator,
            # multiprocessing_context=("forkserver" if len(dataset_wrapped) > 500_000 else "fork"),
        )
        all_corpus_embeddings = embed_dataloader(
            self.model.encoder,
            data_loader, 
            col="text",
            convert_to_tensor=True,
            show_progress_bar=True,
            leave_progress_bar=False,
            output_device="cpu",
            **self.model.model_kwargs
        )
        
        itr = tqdm.trange(0, len(corpus_text), self.corpus_chunk_size, desc=f"Embedding corpus in mini-batches of {self.corpus_chunk_size}")
        result_heaps = {
            qid: [] for qid in query_ids
        }  # Keep only the top-k docs for each query
        for batch_num, corpus_start_idx in enumerate(itr):
            logger.info("Scoring batch {}/{}...".format(batch_num + 1, len(itr)))
            corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(corpus))

            # Encode chunk of corpus
            if (
                self.save_corpus_embeddings
                and "qid" in kwargs
                and len(self.corpus_embeddings[kwargs["qid"]])
            ):
                sub_corpus_embeddings = torch.tensor(
                    self.corpus_embeddings[kwargs["qid"]][batch_num]
                )
            else:
                # Get doc dataset embeddings
                sub_corpus_embeddings = all_corpus_embeddings[corpus_start_idx:corpus_end_idx].cuda()

                if self.save_corpus_embeddings and "qid" in kwargs:
                    self.corpus_embeddings[kwargs["qid"]].append(sub_corpus_embeddings)

            # Compute similarites using either cosine-similarity or dot product
            cos_scores = self.score_functions[score_function](
                query_embeddings, sub_corpus_embeddings
            )
            cos_scores[torch.isnan(cos_scores)] = -1
            # breakpoint()

            # Get top-k values
            cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(
                cos_scores,
                min(
                    top_k + 1,
                    len(cos_scores[1]) if len(cos_scores) > 1 else len(cos_scores[-1]),
                ),
                dim=1,
                largest=True,
                sorted=return_sorted,
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
                self.results[qid][corpus_id] = score

        return self.results

    def load_results_file(self):
        # load the first stage results from file in format {qid: {doc_id: score}}
        if "https://" in self.previous_results:
            # download the file
            if not os.path.exists(self.previous_results):
                url_descriptor = self.previous_results.split("https://")[-1].replace(
                    "/", "--"
                )
                dest_file = os.path.join(
                    "results", f"cached_predictions--{url_descriptor}"
                )
                os.makedirs(os.path.dirname(os.path.abspath(dest_file)), exist_ok=True)
                download(self.previous_results, dest_file)
                logger.info(
                    f"Downloaded the previous results at {self.previous_results} to {dest_file}"
                )
            self.previous_results = dest_file

        with open(self.previous_results, "r") as f:
            previous_results = json.load(f)
        assert isinstance(previous_results, dict)
        assert isinstance(previous_results[list(previous_results.keys())[0]], dict)
        return previous_results

    def predict(self, queries, passages, **kwargs):
        raise NotImplementedError(
            "You must implement a predict method for your reranker model"
        )

    def encode(self, sentences: List[str], **kwargs):
        return self.model.encode(sentences, **kwargs)


def is_dres_compatible(model):
    for method in ["encode_queries", "encode_corpus"]:
        op = getattr(model, method, None)
        if not (callable(op)):
            return False
    return True


def is_cross_encoder_compatible(model):
    op = getattr(model, "predict", None)
    if not (callable(op)):
        return False
    return True


# Adapted from https://github.com/beir-cellar/beir/blob/f062f038c4bfd19a8ca942a9910b1e0d218759d4/beir/retrieval/evaluation.py#L9
class RetrievalEvaluator(Evaluator):
    def __init__(
        self,
        retriever=None,
        k_values: List[int] = [1, 3, 5, 10, 20, 100, 1000],
        score_function: str = "cos_sim",
        first_stage_model=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.is_cross_encoder = False
        if is_cross_encoder_compatible(retriever):
            assert False
            logger.info(
                "The custom predict function of the model will be used if not a SentenceTransformer CrossEncoder"
            )
            self.retriever = DenseRetrievalExactSearch(retriever, **kwargs)
            self.is_cross_encoder = True
        else: 
            assert is_dres_compatible(retriever)
            logger.info(
                "The custom encode_queries and encode_corpus functions of the model will be used"
            )
            self.retriever = DenseRetrievalExactSearch(retriever, first_stage_model=first_stage_model, **kwargs)
        self.k_values = k_values
        self.top_k = (
            max(k_values) if "top_k" not in kwargs else kwargs["top_k"]
        )  # can lower it if reranking
        self.score_function = score_function

    def __call__(
        self, 
        corpus: dict[str, dict[str, str]],
        corpus_cluster_ids: dict[int, List[str]],
        queries: dict[str, str],
        query_cluster_ids: dict[str, int],
    ) -> dict[str, dict[str, float]]:
        if not self.retriever:
            raise ValueError("Model/Technique has not been provided!")

        if self.is_cross_encoder:
            assert False # not currently supported
            return self.retriever.search_cross_encoder(
                corpus, queries, self.top_k
            )
        else:
            return self.retriever.search(
                corpus, 
                corpus_cluster_ids, 
                queries, 
                query_cluster_ids,
                self.top_k, 
                self.score_function
            )

    @staticmethod
    def evaluate(
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        k_values: List[int],
        ignore_identical_ids: bool = True,
    ) -> Tuple[Dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
        if ignore_identical_ids:
            logger.info(
                "For evaluation, we ignore identical query and document ids (default), please explicitly set ``ignore_identical_ids=False`` to ignore this."
            )
            popped = []
            for qid, rels in results.items():
                for pid in list(rels):
                    if qid == pid:
                        results[qid].pop(pid)
                        popped.append(pid)

        ndcg = {}
        _map = {}
        recall = {}
        precision = {}

        for k in k_values:
            ndcg[f"NDCG@{k}"] = 0.0
            _map[f"MAP@{k}"] = 0.0
            recall[f"Recall@{k}"] = 0.0
            precision[f"P@{k}"] = 0.0

        map_string = "map_cut." + ",".join([str(k) for k in k_values])
        ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
        recall_string = "recall." + ",".join([str(k) for k in k_values])
        precision_string = "P." + ",".join([str(k) for k in k_values])
        evaluator = pytrec_eval.RelevanceEvaluator(
            qrels, {map_string, ndcg_string, recall_string, precision_string}
        )
        scores = evaluator.evaluate(results)

        for query_id in scores.keys():
            for k in k_values:
                ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
                _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
                recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
                precision[f"P@{k}"] += scores[query_id]["P_" + str(k)]

        for k in k_values:
            ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"] / len(scores), 5)
            _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"] / len(scores), 5)
            recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"] / len(scores), 5)
            precision[f"P@{k}"] = round(precision[f"P@{k}"] / len(scores), 5)

        for eval in [ndcg, _map, recall, precision]:
            logger.info("\n")
            for k in eval.keys():
                logger.info("{}: {:.4f}".format(k, eval[k]))

        return ndcg, _map, recall, precision

    @staticmethod
    def evaluate_custom(
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        k_values: List[int],
        metric: str,
    ) -> Tuple[Dict[str, float]]:
        if metric.lower() in ["mrr", "mrr@k", "mrr_cut"]:
            return mrr(qrels, results, k_values)
        elif metric.lower() in ["recall_cap", "r_cap", "r_cap@k"]:
            return recall_cap(qrels, results, k_values)
        elif metric.lower() in ["hole", "hole@k"]:
            return hole(qrels, results, k_values)
        elif metric.lower() in [
            "acc",
            "top_k_acc",
            "accuracy",
            "accuracy@k",
            "top_k_accuracy",
        ]:
            return top_k_accuracy(qrels, results, k_values)
