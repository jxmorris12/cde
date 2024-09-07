from typing import List

import json
import logging
import math
import os
import pickle

from collections import defaultdict
from time import time
from typing import Dict, Tuple


import datasets
from datasets import Features, Value, load_dataset
import numpy as np
import torch
import tqdm

from ..evaluation.evaluators import RetrievalEvaluator
from .AbsTask import AbsTask

from cde.lib import get_cde_cache_dir
from cde.lib.cluster import embed_with_cache
from cde.lib.cluster_faiss import paired_kmeans_faiss

logger = logging.getLogger(__name__)


def cluster_corpus_uncached(
        corpus_ds: datasets.Dataset,
        embedding_ds: datasets.Dataset, 
        cluster_size: int
    ) -> Dict[int, List[str]]:
    X = embedding_ds["embeds"]
    # TODO (cache results)
    num_clusters = math.ceil(len(X) / cluster_size)
    _, assignments = paired_kmeans_faiss(
        q=X,
        X=X,
        k=num_clusters,
        max_iters=40,
        n_redo=2,
        seed=42,
    )
    # Compute centroids and per-id assignments.
    centroids = []
    cluster_ids = {}
    for i in range(num_clusters):
        cluster_idxs = np.where(assignments == i)[0]
        centroid = embedding_ds.select(cluster_idxs)["embeds"].mean(dim=0)
        centroids.append(centroid)
        cluster_ids[i] = corpus_ds.select(cluster_idxs)["id"]
    
    centroids = torch.stack(centroids).numpy()
    return centroids, cluster_ids

def cluster_corpus(
        corpus_ds: datasets.Dataset,
        embedding_ds: datasets.Dataset,
        cluster_size: int
    ) -> Tuple[torch.Tensor, Dict[int, List[str]]]:
    cache_dir = os.path.join(
        get_cde_cache_dir(),
        "mteb_cache", 
        "cluster"
    )
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(
        cache_dir,
        f"{corpus_ds._fingerprint}_{cluster_size}_clusters"
    )
    if os.path.exists(cache_file):
        print(f"[AbsTaskRetrieval] Loading cached cluster data from {cache_file}")
        data = pickle.load(open(cache_file, "rb"))
        centroids = data["centroids"]
        cluster_ids = data["cluster_ids"]
    else:
        centroids, cluster_ids = cluster_corpus_uncached(
            corpus_ds, 
            embedding_ds, 
            cluster_size
        )
        pickle.dump(
            {
                "centroids": centroids,
                "cluster_ids": cluster_ids,
            },
            open(cache_file, "wb"),
        )
    return torch.tensor(centroids), cluster_ids

# Adapted from https://github.com/beir-cellar/beir/blob/f062f038c4bfd19a8ca942a9910b1e0d218759d4/beir/datasets/data_loader_hf.py#L10
class HFDataLoader:
    def __init__(
        self,
        hf_repo: str = None,
        hf_repo_qrels: str = None,
        data_folder: str = None,
        prefix: str = None,
        cluster_size: int = 64,
        corpus_file: str = "corpus.jsonl",
        query_file: str = "queries.jsonl",
        qrels_folder: str = "qrels",
        qrels_file: str = "",
        cluster_embedder: str = "",
        streaming: bool = False,
        keep_in_memory: bool = False,
    ):
        self.corpus = {}
        self.queries = {}
        self.qrels = {}

        self.hf_repo = hf_repo
        if hf_repo:
            # By default fetch qrels from same repo not a second repo with "-qrels" like in original
            self.hf_repo_qrels = hf_repo_qrels if hf_repo_qrels else hf_repo
        else:
            # data folder would contain these files:
            # (1) fiqa/corpus.jsonl  (format: jsonlines)
            # (2) fiqa/queries.jsonl (format: jsonlines)
            # (3) fiqa/qrels/test.tsv (format: tsv ("\t"))
            if prefix:
                query_file = prefix + "-" + query_file
                qrels_folder = prefix + "-" + qrels_folder

            self.corpus_file = (
                os.path.join(data_folder, corpus_file) if data_folder else corpus_file
            )
            self.query_file = (
                os.path.join(data_folder, query_file) if data_folder else query_file
            )
            self.qrels_folder = (
                os.path.join(data_folder, qrels_folder) if data_folder else None
            )
            self.qrels_file = qrels_file
        self.cluster_size = cluster_size
        self.streaming = streaming
        self.keep_in_memory = keep_in_memory
        self.cluster_embedder = cluster_embedder

    @staticmethod
    def check(fIn: str, ext: str):
        if not os.path.exists(fIn):
            raise ValueError(
                "File {} not present! Please provide accurate file.".format(fIn)
            )

        if not fIn.endswith(ext):
            raise ValueError(
                "File {} must be present with extension {}".format(fIn, ext)
            )
        
    def load(
        self, split="test"
    ) -> Tuple[Dict[str, dict[str, str]], dict[str, str], dict[str, dict[str, int]]]:
        if not self.hf_repo:
            self.qrels_file = os.path.join(self.qrels_folder, split + ".tsv")
            self.check(fIn=self.corpus_file, ext="jsonl")
            self.check(fIn=self.query_file, ext="jsonl")
            self.check(fIn=self.qrels_file, ext="tsv")

        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d %s Documents.", len(self.corpus), split.upper())
            logger.info("Doc Example: %s", self.corpus[0])

        if not len(self.queries):
            logger.info("Loading Queries...")
            self._load_queries()

        self._load_qrels(split)
        # filter queries with no qrels
        qrels_dict = defaultdict(dict)

        def qrels_dict_init(row):
            qrels_dict[row["query-id"]][row["corpus-id"]] = int(row["score"])

        self.qrels.map(qrels_dict_init)
        self.qrels = qrels_dict
        self.queries = self.queries.filter(lambda x: x["id"] in self.qrels).flatten_indices()
        logger.info("Loaded %d %s Queries.", len(self.queries), split.upper())
        logger.info("Query Example: %s", self.queries[0])

        # queries MIPS
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        query_embeddings = self._embed(self.queries, key="query")["embeds"]
        query_embeddings /= query_embeddings.norm(p=2, dim=1, keepdim=True)
        qe = query_embeddings.to(device)
        ce = self.corpus_cluster_centroids.to(device)
        query_cluster_ids = {}
        for i in tqdm.trange(len(qe), desc="Query MIPS", colour="red", leave=False):
            # Dataset is formatted in a weird way so I have to do this instead of self.queries[i].
            query_id = self.queries.select([i])["id"][0]
            centroid_idx = (qe[i] @ ce.T).argmax(dim=0).item()
            query_cluster_ids[query_id] = centroid_idx

        return (
            self.corpus, 
            self.queries, 
            self.qrels, 
            self.corpus_cluster_ids,
            query_cluster_ids
        )

    def _embed(self, dataset: datasets.Dataset, key: str) -> torch.Tensor:
        # TODO implement true embedding & caching
        # TODO determine the best data structure for this? maybe use faiss?
        embeddings = embed_with_cache(
            self.cluster_embedder,
            dataset._fingerprint + "_" + key, 
            dataset,
            "text",
            save_to_disk=True,
            batch_size=2048,
        )
        assert len(embeddings) == len(dataset)
        return embeddings

    def load_corpus(self) -> dict[str, dict[str, str]]:
        if not self.hf_repo:
            self.check(fIn=self.corpus_file, ext="jsonl")

        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d %s Documents.", len(self.corpus))
            logger.info("Doc Example: %s", self.corpus[0])

        return self.corpus

    def _load_corpus(self):
        if self.hf_repo:
            corpus_ds = load_dataset(
                self.hf_repo,
                "corpus",
                keep_in_memory=self.keep_in_memory,
                streaming=self.streaming,
            )
        else:
            corpus_ds = load_dataset(
                "json",
                data_files=self.corpus_file,
                streaming=self.streaming,
                keep_in_memory=self.keep_in_memory,
            )
        corpus_ds = next(iter(corpus_ds.values()))  # get first split
        corpus_ds = corpus_ds.cast_column("_id", Value("string"))
        corpus_ds = corpus_ds.rename_column("_id", "id")
        corpus_ds = corpus_ds.remove_columns(
            [
                col
                for col in corpus_ds.column_names
                if col not in ["id", "text", "title"]
            ]
        )
        self.corpus = corpus_ds
        corpus_embeddings_ds = self._embed(self.corpus, key="corpus")
        corpus_embeddings = corpus_embeddings_ds["embeds"]
        corpus_embeddings /= corpus_embeddings.norm(p=2, dim=1, keepdim=True)
        self.corpus_embeddings = corpus_embeddings

        self.corpus_cluster_centroids, self.corpus_cluster_ids = (
            cluster_corpus(
                corpus_ds=corpus_ds,
                embedding_ds=corpus_embeddings_ds, 
                cluster_size=self.cluster_size
            )
        )

    def _load_queries(self):
        if self.hf_repo:
            queries_ds = load_dataset(
                self.hf_repo,
                "queries",
                keep_in_memory=self.keep_in_memory,
                streaming=self.streaming,
            )
        else:
            queries_ds = load_dataset(
                "json",
                data_files=self.query_file,
                streaming=self.streaming,
                keep_in_memory=self.keep_in_memory,
            )
        queries_ds = next(iter(queries_ds.values()))  # get first split
        # queries_ds = queries_ds.cast_column("_id", Value("string"))
        queries_ds = queries_ds.rename_column("_id", "id")
        # queries_ds = queries_ds.remove_columns(
        #     [col for col in queries_ds.column_names if col not in ["id", "text"]]
        # )
        self.queries = queries_ds

    def _load_qrels(self, split):
        if self.hf_repo:
            qrels_ds = load_dataset(
                self.hf_repo_qrels,
                keep_in_memory=self.keep_in_memory,
                streaming=self.streaming,
            )[split]
        else:
            qrels_ds = load_dataset(
                "csv",
                data_files=self.qrels_file,
                delimiter="\t",
                keep_in_memory=self.keep_in_memory,
            )
        features = Features(
            {
                "query-id": Value("string"),
                "corpus-id": Value("string"),
                "score": Value("float"),
            }
        )
        qrels_ds = qrels_ds.cast(features)
        self.qrels = qrels_ds


class AbsTaskRetrieval(AbsTask):
    """Abstract class for re-ranking experiments.

    Child-classes must implement the following properties:

    self.corpus: dict[str, dict[str, str]]
        Semantically, it should contain dict[split_name, dict[sample_id, dict[str, str]]]
        E.g. {"test": {"document_one": {"_id": "d1", "title": "title", "text": "text"}}}

    self.queries: dict[str, dict[str, str]]
        Semantically, it should contain dict[split_name, dict[sample_id, str]]
        E.g. {"test": {"q1": "query"}}

    self.relevant_docs: dict[str, dict[str, dict[str, int]]]
        Semantically, it should contain dict[split_name, dict[sample_id, dict[doc_id, score]]]
        E.g.: {"test": {"q1": {"document_one": 1}}}
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_data(self, **kwargs):
        if self.data_loaded:
            return
        self.corpus, self.queries, self.relevant_docs = {}, {}, {}
        self.corpus_cluster_ids = {}
        self.query_cluster_ids = {}
        self.corpus_ds, self.queries_ds = {}, {}
        dataset_path = self.metadata_dict["dataset"]["path"]
        hf_repo_qrels = (
            dataset_path + "-qrels" if "clarin-knext" in dataset_path else None
        )
        for split in kwargs.get("eval_splits", self.metadata_dict["eval_splits"]):
            (corpus, queries, qrels, cluster_ids, query_cluster_ids) = (
                HFDataLoader(
                    hf_repo=dataset_path,
                    hf_repo_qrels=hf_repo_qrels,
                    streaming=False,
                    keep_in_memory=False,
                    cluster_embedder=self.cluster_embedder,
                    cluster_size=self.cluster_size,
                ).load(split=split)
            )
            # Conversion from DataSet
            self.corpus_ds[split], self.queries_ds[split] = corpus, queries
            queries = dict(zip(queries["id"], queries["text"]))
            corpus = {
                doc["id"]: {"title": doc["title"], "text": doc["text"]}
                for doc in corpus
            }
            self.corpus[split], self.queries[split], self.relevant_docs[split] = (
                corpus,
                queries,
                qrels,
            )
            self.corpus_cluster_ids[split] = cluster_ids
            self.query_cluster_ids[split] = query_cluster_ids

        self.data_loaded = True

    def evaluate(self, model, split="test", first_stage_model=None, **kwargs):
        retriever = RetrievalEvaluator(
            model,
            first_stage_model=first_stage_model,
            **kwargs
        )

        scores = {}
    
        corpus, queries, relevant_docs = (
            self.corpus[split],
            self.queries[split],
            self.relevant_docs[split],
        )
        corpus_cluster_ids = self.corpus_cluster_ids[split]
        query_cluster_ids = self.query_cluster_ids[split]

        scores = self._evaluate_split(
            retriever, 
            corpus, 
            corpus_cluster_ids, 
            queries, 
            query_cluster_ids, 
            relevant_docs, 
            None, 
            **kwargs
        )
        return scores

    def _evaluate_split(
        self, 
        retriever,
        corpus, 
        corpus_cluster_ids,
        queries, 
        query_cluster_ids,
        relevant_docs, 
        lang=None, **kwargs
    ):
        start_time = time()
        results = retriever(corpus, corpus_cluster_ids, queries, query_cluster_ids)
        end_time = time()
        logger.info(
            "Time taken to retrieve: {:.2f} seconds".format(end_time - start_time)
        )

        if kwargs.get("save_predictions", False):
            output_folder = kwargs.get("output_folder", "results")
            if not os.path.isdir(output_folder):
                os.makedirs(output_folder)
            top_k = kwargs.get("top_k", None)
            if top_k is not None:
                for qid in list(results.keys()):
                    doc_ids = set(
                        sorted(
                            results[qid], key=lambda x: results[qid][x], reverse=True
                        )[:top_k]
                    )
                    results[qid] = {
                        k: v for k, v in results[qid].items() if k in doc_ids
                    }
            if lang is None:
                qrels_save_path = (
                    f"{output_folder}/{self.metadata_dict['name']}_predictions.json"
                )
            else:
                qrels_save_path = f"{output_folder}/{self.metadata_dict['name']}_{lang}_predictions.json"

            with open(qrels_save_path, "w") as f:
                json.dump(results, f)

        ndcg, _map, recall, precision = retriever.evaluate(
            relevant_docs,
            results,
            retriever.k_values,
            ignore_identical_ids=kwargs.get("ignore_identical_ids", True),
        )
        mrr = retriever.evaluate_custom(
            relevant_docs, results, retriever.k_values, "mrr"
        )
        scores = {
            **{f"ndcg_at_{k.split('@')[1]}": v for (k, v) in ndcg.items()},
            **{f"map_at_{k.split('@')[1]}": v for (k, v) in _map.items()},
            **{f"recall_at_{k.split('@')[1]}": v for (k, v) in recall.items()},
            **{f"precision_at_{k.split('@')[1]}": v for (k, v) in precision.items()},
            **{f"mrr_at_{k.split('@')[1]}": v for (k, v) in mrr.items()},
        }
        return scores

    def calculate_metadata_metrics(self) -> None:
        self.load_data()

        for split in self.metadata_dict["eval_splits"]:
            if self.is_multilingual:
                for lang in self.relevant_docs.keys():
                    process_language(
                        self.relevant_docs[lang][split],
                        self.queries[lang][split],
                        self.corpus[lang][split],
                        lang,
                    )
            else:
                process_language(
                    self.relevant_docs[split], self.queries[split], self.corpus[split]
                )


def process_language(relevant_docs, queries, corpus, lang=None):
    total_length, num_pairs = calculate_length_and_count(relevant_docs, queries, corpus)
    average_length = total_length / num_pairs if num_pairs else 0
    num_documents = len(queries) + len(corpus)

    language_description = f" for language {lang}" if lang else ""
    print(f"Average character length{language_description} is {average_length}")
    print(f"Number of queries and documents{language_description} is {num_documents}")


def calculate_length_and_count(relevant_docs, queries, corpus):
    total_length = 0
    num_pairs = 0
    for query_id, docs in relevant_docs.items():
        query = queries[query_id]
        for doc_id in docs:
            # not relevant
            if docs[doc_id] == 0:
                continue
            doc = corpus[doc_id]
            doc_text = doc["title"] + doc["text"]
            total_length += len(query) + len(doc_text)
            num_pairs += 1
    return total_length, num_pairs
