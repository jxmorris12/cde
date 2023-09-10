from typing import Any, Dict, List, Optional, Tuple, Union

import collections
import dataclasses
import json
import hashlib
import logging
import os
import pickle
import random
import requests
import zipfile

import datasets
import numpy as np
import torch
import tqdm
import transformers


def get_dataset_name(d: datasets.Dataset) -> str:
    return f"{d.builder_name}.{d.config_name}[{d.split}]"


def process_qrels_uncached(corpus: datasets.Dataset, qrels: datasets.Dataset) -> Tuple[Dict[str, List[float]], Dict[str, List[str]]]:
    qrels_idxs = collections.defaultdict(list)
    qrels_scores = collections.defaultdict(list)
    corpus_ids = np.array(corpus['_id'])
    skipped_qrels = 0

    for ex in tqdm.tqdm(qrels, desc='processing qrels', colour='#964B00', leave=False):
        # example:
        # {
        #  'query-id': 1, 
        #  'corpus-id': 'b0680508-2019-04-18T13:48:51Z-00002-000',
        #  'score': 2
        # }
        # 
        q_id = str(ex['query-id'])
        c_idxs = (corpus_ids == str(ex['corpus-id'])).nonzero()[0]
        # 
        assert len(c_idxs) <= 1, f"error - duplicate corpus ID? (found {len(c_idxs)} matches)"
        # 
        if len(c_idxs):
            qrels_idxs[q_id].append(c_idxs[0])
            qrels_scores[q_id].append(ex['score'])
        else:
            skipped_qrels += 1
        #
    
    if skipped_qrels > 0:
        print(f'Warning: Skipped {skipped_qrels}/{len(qrels)} qrels.')
    
    return qrels_idxs, qrels_scores


def process_qrels(
        corpus: datasets.Dataset, qrels: datasets.Dataset, 
        use_cache: bool = True
    ) -> Tuple[Dict[str, List[float]], Dict[str, List[str]]]:
    dataset_cache_file = '_'.join(
        (corpus.cache_files[0]['filename'], qrels.cache_files[0]['filename'])
    )
    cache_file = strip_extension(dataset_cache_file) + '_processed_qrels.p'
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)

    if not (use_cache and os.path.exists(cache_file)):
        qrels_idxs, qrels_scores = process_qrels_uncached(
            corpus=corpus, qrels=qrels
        )
        if use_cache:
            pickle.dump((qrels_idxs, qrels_scores), open(cache_file, 'wb'))
    else:
        qrels_idxs, qrels_scores = pickle.load(open(cache_file, 'rb'))
    
    return qrels_idxs, qrels_scores


def compute_token_counts_uncached(
        tokenizer: transformers.AutoTokenizer,
        dataset: datasets.Dataset,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    print(f'computing dataset token counts for: {get_dataset_name(dataset)}')

    vocab_size = tokenizer.vocab_size
    token_freqs = torch.zeros(
        (vocab_size, ), dtype=int, requires_grad=False
    )
    document_token_freqs = torch.zeros(
        (vocab_size, ), dtype=int, requires_grad=False
    )

    for ex in tqdm.tqdm(dataset, leave=False, colour='#E6E6FA'):
        ex_valid_keys = ('document_input_ids' in ex) ^ ('text_input_ids' in ex)
        assert ex_valid_keys, f'cannot compute histogram, got ex with keys {ex.keys()}'
        key_name = 'document_input_ids' if ('document_input_ids' in ex) else 'text_input_ids'
        input_ids = ex[key_name]
        assert isinstance(input_ids, torch.Tensor), f'invalid input_ids {type(input_ids)}'
        f = input_ids.bincount(minlength=vocab_size)
        token_freqs += f
        document_token_freqs += f.clamp(max=1)

    return token_freqs, document_token_freqs


def strip_extension(filename: str) -> str:
    """Strips file extension.

    Ex:
        >> strip_extension('/root/dir/sub/file.ext')
        '/root/dir/sub/file'
    """
    return os.path.splitext(filename)[0]


def md5_hash(t: Tuple[str]) -> str:
    return hashlib.md5('__'.join(t).encode()).hexdigest()


def md5_hash_kwargs(**kwargs) -> str:
    # We ignore special hf args that start with _ like '__cached__setup_devices'.
    safe_kwargs = {k: str(v) for k,v in kwargs.items() if not k.startswith('_')}
    s = json.dumps(safe_kwargs, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()

def download_url(url: str, save_path: str, chunk_size: int = 1024):
    """Download url with progress bar using tqdm
    https://stackoverflow.com/questions/15644964/python-progress-bar-and-downloads
    Args:
        url (str): downloadable url
        save_path (str): local path to save the downloaded file
        chunk_size (int, optional): chunking of files. Defaults to 1024.
    """
    r = requests.get(url, stream=True)
    total = int(r.headers.get('Content-Length', 0))
    with open(save_path, 'wb') as fd, tqdm.tqdm(
        desc=save_path,
        total=total,
        unit='iB',
        unit_scale=True,    
        unit_divisor=chunk_size,
    ) as bar:
        for data in r.iter_content(chunk_size=chunk_size):
            size = fd.write(data)
            bar.update(size)


def unzip(zip_file: str, out_dir: str):
    zip_ = zipfile.ZipFile(zip_file, "r")
    zip_.extractall(path=out_dir)
    zip_.close()


def download_url_and_unzip(url: str, out_dir: str, chunk_size: int = 1024) -> str:
    os.makedirs(out_dir, exist_ok=True)
    dataset = url.split("/")[-1]
    zip_file = os.path.join(out_dir, dataset)
    
    if not os.path.isfile(zip_file):
        logging.info("Downloading {} ...".format(dataset))
        download_url(url, zip_file, chunk_size)
    
    if not os.path.isdir(zip_file.replace(".zip", "")):
        logging.info("Unzipping {} ...".format(dataset))
        unzip(zip_file, out_dir)
    
    return os.path.join(out_dir, dataset.replace(".zip", ""))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class RerankHelper:
    """Wraps our model and does reranking.
    
    Template: https://github.com/beir-cellar/beir/blob/main/beir/reranking/rerank.py#L7
    """
    def __init__(self, model):
        self.model = model
    
    def _score(self, query_embedding: str, corpus_embeddings: List[str]) -> float:
        with torch.no_grad():
            scores = self.model(query_embedding=query_embedding[None], document_embeddings=corpus_embeddings[None])
        return scores.flatten().cpu().tolist()
    
    def rerank(self, 
               corpus: Dict[str, Dict[str, str]], 
               corpus_embeddings: datasets.Dataset,
               queries: Dict[str, str],
               query_embeddings: datasets.Dataset,
               results: Dict[str, Dict[str, float]],
               top_k: int) -> Dict[str, Dict[str, float]]:
        
        sentence_pairs, pair_ids = [], []
        rerank_scores_model = []
        rerank_scores_biencoder = []

        query_idx_dict = {key: j for j, key in enumerate(queries['id'])} # Map IDs to idxs
        corpus_idx_dict = {key: j for j, key in enumerate(corpus['id'])} # Map IDs to idxs
        
        #### Starting to Rerank using cross-attention
        logging.info("Starting To Rerank Top-{}....".format(top_k))
        
        for query_id in tqdm.tqdm(results, desc="evaluating dataset"):
            # TODO: call self._score in batch :)
            minicorpus = []
            if len(results[query_id]) > top_k:
                for (doc_id, _) in sorted(results[query_id].items(), key=lambda item: item[1], reverse=True)[:top_k]:
                    pair_ids.append([query_id, doc_id])
                    minicorpus.append(corpus_embeddings[corpus_idx_dict[doc_id]]["embeds"])
            else:
                for doc_id in results[query_id]:
                    pair_ids.append([query_id, doc_id])
                    minicorpus.append(corpus_embeddings[corpus_idx_dict[doc_id]]["embeds"])
            
            query_embedding = torch.tensor(query_embeddings[query_idx_dict[query_id]]["embeds"]).to(device)
            minicorpus_embeddings = torch.tensor(minicorpus).to(device)

            model_score = self._score(query_embedding, minicorpus_embeddings)
            biencoder_score = torch.nn.functional.cosine_similarity(query_embedding, minicorpus_embeddings).flatten().cpu().tolist()
            rerank_scores_model.extend(model_score)
            rerank_scores_biencoder.extend(biencoder_score)

        #### Reranking results
        rerank_results_model = {query_id: {} for query_id in results}
        rerank_results_biencoder = {query_id: {} for query_id in results}
        for pair, model_score, biencoder_score in zip(pair_ids, rerank_scores_model, rerank_scores_biencoder):
            query_id, doc_id = pair[0], pair[1]
            rerank_results_model[query_id][doc_id] = model_score
            rerank_results_biencoder[query_id][doc_id] = biencoder_score

        return (rerank_results_model, rerank_results_biencoder) 