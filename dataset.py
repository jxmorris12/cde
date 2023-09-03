from typing import Any, Dict, List, Optional, Tuple
import gzip
import json
import logging
import os
import random

import beir.datasets.data_loader
import datasets
import numpy as np
import pickle
import torch
import transformers
import tqdm
from helpers import (
    download_url_and_unzip, download_url, tokenize_dataset
)

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


def load_msmarco_beir_uncached(split: str) -> Tuple[datasets.Dataset, datasets.Dataset]:
    """Loads MSMARCO through tools provided by BeIR.

    Returns:
        corpus (datasets.Dataset): Corpus of documents
            keys -- corpus_id, text
        queries (datasets.Dataset):  Corpus of queries
            keys -- query_id, text
    """

    #### Download msmarco.zip dataset and unzip the dataset
    dataset = "msmarco"
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
    data_path = download_url_and_unzip(url, out_dir)

    ### Load BEIR MSMARCO training dataset, this will be used for query and corpus for reference.
    corpus, queries, _ = beir.datasets.data_loader.GenericDataLoader(data_path).load(split=split)

    corpus = datasets.Dataset.from_list(
        [{"id": k, "text": v["text"]} for k,v in corpus.items()])
    # corpus._fingerprint = md5_hash(f"msmarco_beir_{split}") 

    queries = datasets.Dataset.from_list([{ "id": k, "text": v} for k,v in queries.items()])
    # queries._fingerprint = md5_hash(f"msmarco_beir_{split}") 

    return corpus, queries


def embed_with_cache(embedder: str, cache_name: str, texts: List[str]) -> datasets.Dataset:
    embedder_cache_path = embedder.replace('/', '__')
    # cache_folder = datasets.config.HF_DATASETS_CACHE
    cache_folder = "/scratch/jxm3"
    cache_folder = os.path.join(cache_folder, 'corpus_embeddings', embedder_cache_path)
    os.makedirs(cache_folder, exist_ok=True)
    cache_path = os.path.join(cache_folder, cache_name) #  + "_small")

    # texts = texts[:1000]

    print("saving/loading embeddings at path:", cache_path)

    if os.path.exists(cache_path):
        return datasets.load_from_disk(cache_path)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(embedder)
    model.max_seq_length = 512
    embeddings = model.encode(texts, show_progress_bar=True)
    # import numpy as np
    # embeddings = np.random.rand(len(texts), 768)

    datasets_list = []
    dataset_size = 1_000_000
    i = 0
    while i < len(embeddings):
        dataset = datasets.Dataset.from_dict({
            "embeds": embeddings[i : i + dataset_size] 
        })
        datasets_list.append(dataset)
        i += dataset_size
    
    return datasets.concatenate_datasets(datasets_list)


def load_msmarco_beir(split: str) -> Tuple[datasets.Dataset, datasets.Dataset]:
    cache_path = datasets.config.HF_DATASETS_CACHE # something like /home/jxm3/.cache/huggingface/datasets
    corpus_path = os.path.join(cache_path, f'tti_beir_local_msmarco_corpus_{split}')
    queries_path = os.path.join(cache_path, f'tti_beir_local_msmarco_queries_{split}')

    print(corpus_path)

    if os.path.exists(corpus_path) and os.path.exists(queries_path):
        logging.info("Loading MSMARCO split %s corpus from path %s", split, corpus_path)
        corpus = datasets.load_from_disk(corpus_path)
        logging.info("Loading MSMARCO split %s queries from path %s", split, queries_path)
        queries = datasets.load_from_disk(queries_path)
    else:
        corpus, queries = load_msmarco_beir_uncached(split=split)
        corpus.save_to_disk(corpus_path)
        logging.info("Saving MSMARCO split %s corpus to path %s", split, corpus_path)
        queries.save_to_disk(queries_path)
        logging.info("Saving MSMARCO split %s queries to path %s", split, queries_path)
    
    return corpus, queries


class MSMarcoDataset(torch.utils.data.Dataset):
    corpus: datasets.Dataset
    queries: datasets.Dataset
    size: int
    tokenized: bool
    token_histogram: torch.Tensor
    token_idf: torch.Tensor
    column_names: List[str] = ["id", "text_input_ids", "text_attention_mask"]
    hard_negatives: Optional[Dict[str, Any]]
    def __init__(
            self,
            embedder: str
        ):
       
        self.corpus, self.queries = load_msmarco_beir(split="train")
        self.query_embeddings = embed_with_cache(embedder, "msmarco_queries", [q['text'] for q in self.queries])
        self.corpus_embeddings = embed_with_cache(embedder, "msmarco_corpus", [c['text'] for c in self.corpus])
        self.hard_negatives = load_msmarco_hard_negatives()
        self.tokenized = False
        self.size = len(self.query_embeddings)
    
    def __len__(self):
        return self.size

    def next_dataset_idx(self) -> int:
        return 0
        
    def reset_dataset_idx(self, idx: int) -> None:
        pass

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Returns example from MSMARCO, including query, document, and hard-negative document."""
        q_ex = self.queries[idx] # %1000 is temp (testing)

        query_id_str = q_ex['id']

        ex = {}

        hn_dict = self.hard_negatives[query_id_str]
        document_id = int(hn_dict['pos'][0])

        # neg = self.corpus[int(random.choice(hn_dict['hard_neg']))]
        # ex['hn_document_input_ids'] = neg['text_input_ids']
        # ex['hn_document_attention_mask'] = neg['text_attention_mask']

        return {
            "query_embedding": self.query_embeddings[int(query_id_str)]["embeds"],
            "document_embeddings": self.corpus_embeddings[document_id]["embeds"],
        }


if __name__ == '__main__':
    dataset = MSMarcoDataset(embedder="sentence-transformers/gtr-t5-base")