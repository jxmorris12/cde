from typing import Any, Dict, List, Optional, Tuple
import glob
import gzip
import json
import logging
import os
import pathlib
import random

import beir.datasets.data_loader
import datasets
import numpy as np
import pickle
import random
import torch
import transformers
import tqdm
from helpers import (
    download_url_and_unzip, download_url, md5_hash
)

def datasets_fast_load_from_disk(cache_path) -> datasets.Dataset:
    # files = glob.glob(os.path.join(cache_path, "*.arrow"))
    # files_tqdm = tqdm.tqdm(files, desc=f"loading from {cache_path}", leave=False, colour="#FFC0CB")
    # datasets_list = [datasets.Dataset.from_file(file) for file in files_tqdm]
    # return datasets.concatenate_datasets(datasets_list)
    return datasets.load_from_disk(cache_path)


def tokenize_dataset(
        dataset: datasets.Dataset,
        tokenizer: transformers.PreTrainedTokenizer,
        max_length: int,
        text_key: str,
        padding_strategy: str
    ) -> datasets.Dataset:
    def tokenize_text(ex: Dict) -> Dict:
        tt = tokenizer(
            ex[text_key],
            max_length=max_length,
            truncation=True,
            padding=padding_strategy,
        )
        for k,v in tt.items():
            ex[f"{text_key}_{k}"] = v
        ex["length"] = [len(tt) for tt in ex[f"{text_key}_input_ids"]]
        return ex

    # generate unique hash for tokenizer
    vocab = tokenizer.vocab
    vocab_words = tuple(sorted(vocab.keys(), key=lambda word: vocab[word]))
    vocab_hash = md5_hash(vocab_words)

    data_fingerprint = '__'.join((
        dataset._fingerprint, str(vocab_hash), str(max_length),
        text_key, padding_strategy
    ))
    data_fingerprint = md5_hash(data_fingerprint)
    dataset = dataset.map(
        tokenize_text,
        new_fingerprint=data_fingerprint,
        batched=True,
        load_from_cache_file=True,
    )
    return dataset

def load_msmarco_hard_negatives_uncached() -> Dict[str, Dict[str, Any]]:
    """Loads hard negative passage for MSMARCO.

    Adapted from github.com/beir-cellar/beir/blob/main/examples/retrieval/training/train_msmarco_v3.py
    """
    ce_score_margin = 3             # Margin for the CrossEncoder score between negative and positive passages
    num_negs_per_system = 5         # We used different systems to mine hard negatives. Number of hard negatives to add from each system

    triplets_url = "https://sbert.net/datasets/msmarco-hard-negatives.jsonl.gz"

    msmarco_triplets_filepath = os.path.join("datasets", "msmarco-hard-negatives.jsonl.gz")
    if not os.path.isfile(msmarco_triplets_filepath):
        download_url(triplets_url, msmarco_triplets_filepath)

    #### Load the hard negative MSMARCO jsonl triplets from SBERT 
    #### These contain a ce-score which denotes the cross-encoder score for the query and passage.
    #### We chose a margin between positive and negative passage scores => above which consider negative as hard negative. 
    #### Finally to limit the number of negatives per passage, we define num_negs_per_system across all different systems.

    logging.info("Loading MSMARCO hard-negatives...")

    hn = {}
    with gzip.open(msmarco_triplets_filepath, 'rt', encoding='utf8') as fIn:
        for line in tqdm.tqdm(fIn, total=502_939):
            data = json.loads(line)
            
            # Get the positive passage ids
            pos_pids = [item['pid'] for item in data['pos']]
            pos_min_ce_score = min([item['ce-score'] for item in data['pos']])
            ce_score_threshold = pos_min_ce_score - ce_score_margin
            
            # Get the hard negatives
            neg_pids = set()
            for system_negs in data['neg'].values():
                negs_added = 0
                for item in system_negs:
                    if item['ce-score'] > ce_score_threshold:
                        continue

                    pid = item['pid']
                    if pid not in neg_pids:
                        neg_pids.add(pid)
                        negs_added += 1
                        if negs_added >= num_negs_per_system:
                            break
            
            if len(pos_pids) > 0 and len(neg_pids) > 0:
                hn[data['qid']] = {
                    'pos': pos_pids,
                    'hard_neg': list(neg_pids)
                }

    return hn


def load_msmarco_hard_negatives() -> Dict[str, Dict[str, Any]]:
    cache_file = os.path.join("datasets", "msmarco-hard-negatives-processed.p")
    if not os.path.isfile(cache_file):
        train_queries = load_msmarco_hard_negatives_uncached()
        pickle.dump(train_queries, open(cache_file, 'wb'))
    else:
        train_queries = pickle.load(open(cache_file, 'rb'))
    
    return train_queries


def get_bm25_results(dataset: str, corpus, queries) -> Dict:
    from beir.retrieval.evaluation import EvaluateRetrieval
    from beir.retrieval.search.lexical import BM25Search as BM25

    index_name = f"beir_{dataset}"
    username = "elastic"
    password = "FjZD_LI-=AJOtsfpq9U*"

    # url = f"https://{username}:{password}@rush-compute-01.tech.cornell.edu:9200""
    hostname = f"{username}:{password}@rush-compute-01.tech.cornell.edu:9200"
    bm25_model = BM25(index_name=index_name, hostname=hostname, initialize=True)
    retriever = EvaluateRetrieval(bm25_model)
    return retriever.retrieve(corpus, queries)


def get_ance_results(dataset: str, corpus, queries) -> Dict:
    if len(corpus) > 100_000:
        print(f"Auto-skipping ANCE evaluation of {dataset} -- corpus too large.")
        return {}
    from beir.retrieval import models
    from beir.retrieval.evaluation import EvaluateRetrieval
    from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
    model = DRES(models.SentenceBERT("msmarco-roberta-base-ance-firstp"), batch_size=16, corpus_chunk_size=512*2)
    retriever = EvaluateRetrieval(model, score_function="dot")
    return retriever.retrieve(corpus, queries)


def load_beir_uncached(dataset: str, split: str) -> Tuple[datasets.Dataset, datasets.Dataset, Dict[str, Dict[str, int]], Dict]:
    """Loads a BEIR test dataset through tools provided by BeIR.

    Returns:
        corpus (datasets.Dataset): Corpus of documents
            keys -- corpus_id, text
        queries (datasets.Dataset):  Corpus of queries
            keys -- query_id, text
        qrels
        ance_results
    """

    print("loading dataset uncached", dataset)
    #### Download msmarco.zip dataset and unzip the dataset
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
    print("... downloading")
    data_path = download_url_and_unzip(url, out_dir)

    print("... loading split", split, "at path", data_path)
    ### Load BEIR MSMARCO training dataset, this will be used for query and corpus for reference.
    corpus, queries, qrels = beir.datasets.data_loader.GenericDataLoader(data_path).load(split=split)
    # bm25_results = get_bm25_results(dataset=dataset, corpus=corpus, queries=queries)
    print("... getting ance results")
    ance_results = get_ance_results(dataset=dataset, corpus=corpus, queries=queries)

    corpus = datasets.Dataset.from_list(
        [{"id": k, "text": v["text"]} for k,v in corpus.items()])
    # corpus._fingerprint = md5_hash(f"msmarco_beir_{split}") 

    queries = datasets.Dataset.from_list([{ "id": k, "text": v} for k,v in queries.items()])
    # queries._fingerprint = md5_hash(f"msmarco_beir_{split}") 

    return corpus, queries, qrels, ance_results


def embed_with_cache(embedder: str, cache_name: str, texts: List[str]) -> datasets.Dataset:
    print("embed_with_cache ...")
    embedder_cache_path = embedder.replace('/', '__')
    # cache_folder = datasets.config.HF_DATASETS_CACHE
    cache_folder = "/scratch/jxm3"
    cache_folder = os.path.join(cache_folder, 'corpus_embeddings', embedder_cache_path)
    os.makedirs(cache_folder, exist_ok=True)
    cache_path = os.path.join(cache_folder, cache_name) #  + "_small")

    # texts = texts[:1000]


    if os.path.exists(cache_path):
        print("[embed_with_cache] Loading embeddings at path:", cache_path)
        return datasets_fast_load_from_disk(cache_path)

    print("[embed_with_cache] computing embeddings to save at path:", cache_path)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(embedder)
    model.max_seq_length = 512
    embeddings = model.encode(texts, show_progress_bar=True)
    # import numpy as np
    # embeddings = np.random.rand(len(texts), 768)

    datasets_list = []
    max_dataset_size = 1_000_000
    i = 0
    while i < len(embeddings):
        dataset = datasets.Dataset.from_dict({
            "embeds": embeddings[i : i + max_dataset_size] 
        })
        datasets_list.append(dataset)
        i += max_dataset_size
    
    d = datasets.concatenate_datasets(datasets_list)
    d.save_to_disk(cache_path)
    return d


def load_beir(dataset: str, split: str) -> Tuple[datasets.Dataset, datasets.Dataset, Dict[str, Dict[str, int]]]:
    print("loading dataset", dataset)
    cache_path = datasets.config.HF_DATASETS_CACHE # something like /home/jxm3/.cache/huggingface/datasets
    corpus_path = os.path.join(cache_path, f'tti_beir_local_{dataset}_corpus_{split}')
    queries_path = os.path.join(cache_path, f'tti_beir_local_{dataset}_queries_{split}')
    qrels_path = os.path.join(cache_path, f'tti_beir_local_{dataset}_qrels_{split}.p')
    # bm25_path = os.path.join(cache_path, f'tti_beir_local_{dataset}_bm25_results')
    ance_path = os.path.join(cache_path, f'tti_beir_local_{dataset}_ance_results')

    if os.path.exists(corpus_path) and os.path.exists(queries_path) and os.path.exists(qrels_path) and os.path.exists(ance_path):
        logging.info(f"Loading {dataset} split %s corpus from path %s", split, corpus_path)
        corpus = datasets_fast_load_from_disk(corpus_path)
        logging.info(f"Loading {dataset} split %s queries from path %s", split, queries_path)
        queries = datasets_fast_load_from_disk(queries_path)
        logging.info(f"Loading {dataset} split %s qrels from path %s", split, qrels_path)
        qrels = pickle.load(open(qrels_path, 'rb'))
        logging.info(f"Loading {dataset} split %s ance results from path %s", split, ance_path)
        ance_results = pickle.load(open(ance_path, 'rb'))
    else:
        corpus, queries, qrels, ance_results = load_beir_uncached(dataset=dataset, split=split)
        logging.info(f"Saving {dataset} split %s corpus to path %s", split, corpus_path)
        corpus.save_to_disk(corpus_path)
        logging.info(f"Saving {dataset} split %s queries to path %s", split, queries_path)
        queries.save_to_disk(queries_path)
        pickle.dump(qrels, open(qrels_path, 'wb'))
        pickle.dump(ance_results, open(ance_path, 'wb'))

    return corpus, queries, qrels, ance_results


class BeirDataset(torch.utils.data.Dataset):
    name: str
    corpus: datasets.Dataset
    queries: datasets.Dataset
    query_embeddings: datasets.Dataset
    corpus_embeddings: datasets.Dataset
    ance_results: Dict[str, Dict[str, int]]
    size: int
    column_names: List[str] = ["idx", "query_embedding", "document_embeddings", "negative_document_embeddings"]
    hard_negatives: Optional[Dict[str, Any]]
    def __init__(
            self,
            dataset: str,
            embedder: str,
            split: str = "test"
        ):
        self.name = dataset
        self.corpus, self.queries, self.qrels, self.ance_results = load_beir(dataset=dataset, split=split)
        print(f">> embedding dataset {dataset} split {split}")
        self.query_embeddings = embed_with_cache(
            embedder, f"{dataset}_queries" + ("" if split == "train" else f"_{split}"), 
            self.queries["text"],
        )
        self.corpus_embeddings = embed_with_cache(
            embedder, 
            f"{dataset}_corpus" + ("" if split == "train" else f"_{split}"), 
            self.corpus["text"],
        )
        self.size = len(self.queries)
    
    def __len__(self) -> int:
        return self.size
    
    def tokenize(self, tokenizer: transformers.PreTrainedTokenizer, max_length: int) -> None:
        print(f"tokenizing {self.name}")
        self.queries = tokenize_dataset(
            dataset=self.queries,
            tokenizer=tokenizer,
            max_length=max_length,
            text_key="text",
            padding_strategy="do_not_pad"
        )
        self.corpus = tokenize_dataset(
            dataset=self.corpus,
            tokenizer=tokenizer,
            max_length=max_length,
            text_key="text",
            padding_strategy="do_not_pad"
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Returns example from BEIR, including query, document, and hard-negative document."""
        ex = {}
        if idx < len(self.queries):
            ex.update(self.queries[idx])
            q_id = self.query_ids[idx]
            ex['qrels_idxs'] = self.qrels_idxs[q_id]
            ex['qrels_scores'] = self.qrels_scores[q_id]
            if not len(ex['qrels_idxs']):
                print('Warning: BEIR dataset trying to return example without qrels')
                # raise ValueError('BEIR dataset trying to return example without qrels')
        # if idx < len(self.corpus):
        #     ex.update({ "self.corpus_embedding": }[idx])
        # assert ('document_input_ids' in ex) or ('query_input_ids' in ex)
        # return ex
        
        return {
            "idx": idx,
            "query_embedding": self.query_embeddings[idx]["embeds"],
            "document_embeddings": self.corpus_embeddings[document_id]["embeds"],
            "negative_document_embeddings": self.corpus_embeddings[neg_embedding_id]["embeds"],
        }


class MsmarcoDatasetHardNegatives(BeirDataset):
    corpus: datasets.Dataset
    queries: datasets.Dataset
    query_embeddings: datasets.Dataset
    corpus_embeddings: datasets.Dataset
    size: int
    column_names: List[str] = ["idx", "query_embedding", "document_embeddings", "negative_document_embeddings"]
    hard_negatives: Optional[Dict[str, Any]]
    def __init__(
            self,
            embedder: str
        ):
        super().__init__(dataset="msmarco", split="train", embedder=embedder)
        self.hard_negatives = load_msmarco_hard_negatives()

    def __getitem__(self, query_id: int) -> Dict[str, torch.Tensor]:
        """Returns example from MSMARCO, including query, document, and hard-negative document."""
        query_id_str = self.queries[query_id]['id']
        while query_id_str not in self.hard_negatives:
            # We don't have negative samples for a few queries in the corpus 
            # (maybe because the hard negatives were filtered out by the 
            # cross encoder?). This iterates until we find one.
            #  TODO: Figure out why some queries are missing.
            query_id = random.randint(0, len(self)-1)
            query_id_str = self.queries[query_id]['id']

        hn_dict = self.hard_negatives[query_id_str]
        
        ex = {}

        ex['query_input_ids'] = self.queries[query_id]['text_input_ids']
        ex['query_attention_mask'] = self.queries[query_id]['text_attention_mask']

        pos_doc_id = int(hn_dict['pos'][0])
        ex['document_input_ids'] = self.corpus[pos_doc_id]['text_input_ids']
        ex['document_attention_mask'] = self.corpus[pos_doc_id]['text_attention_mask']

        neg_doc_id = int(random.choice(hn_dict['hard_neg']))
        ex['negative_document_input_ids'] = self.corpus[neg_doc_id]['text_input_ids']
        ex['negative_document_attention_mask'] = self.corpus[neg_doc_id]['text_attention_mask']

        ex.update({
            "idx": query_id,
            "query_embedding": self.query_embeddings[query_id]["embeds"],
            "document_embeddings": self.corpus_embeddings[pos_doc_id]["embeds"],
            "negative_document_embeddings": self.corpus_embeddings[neg_doc_id]["embeds"],
        })
        return ex


if __name__ == '__main__':
    nfcorpus = BeirDataset(
        dataset="nfcorpus",    
        embedder="sentence-transformers/gtr-t5-base"
    )

    dataset = MsmarcoDatasetHardNegatives(
        embedder="sentence-transformers/gtr-t5-base"
    )
    print(dataset[10_001])
