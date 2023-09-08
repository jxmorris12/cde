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


# def inverse_cloze(
#     input_ids: torch.Tensor, pad_token_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
#     """Creates Inverse Cloze ..."""


def independent_crop(
    input_ids: torch.Tensor, pad_token_id: int,
    l1: int = 256, l2: int = 256) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns two independent crops from input_ids.
    
    Args:
        input_ids: tensor of IDs
        pad_token_id: ID of pad tokens in input_ids
        l1: length of span 1, cropped
        l2: length of span 2, cropped
    Returns:
        span1: first crop (of length l1)
        span2: second crop (of length l2)
    """ 
    # Count tokens until pad.
    if (input_ids == pad_token_id).sum() == 0:
        N = len(input_ids)
    else:
        N = (input_ids == pad_token_id).int().argmax().item()
    
    ####
    ###
    ##
    ## Contriever:  We use the random cropping data
    ## augmentation, with documents of 256 tokens and span 
    ## sizes sampled between 5% and 50% of the document
    ## length
    ##
    ###
    #####
    ####### LaPraDor: The maximum lengths set for queries and
    ####### documents are 64 and 350...
    #####
    # TODO is this divide-by-two a good idea? (Don't want s1=s2 ever..)
    nl1 = min(N//2, l1)
    nl2 = min(N//2, l2)

    s1_start = random.randint(0, N-nl1)
    s2_start = random.randint(0, N-nl2)

    s1 = input_ids[s1_start:s1_start+nl1]
    s2 = input_ids[s2_start:s2_start+nl2]
    # if len(s1) < l1:
    #     p1 = torch.full(size=(l1 - len(s1),), fill_value=pad_token_id).to(s1.device)
    #     s1 = torch.cat((s1, p1), dim=0)
    # if len(s2) < l2:
    #     p2 = torch.full(size=(l2 - len(s2),), fill_value=pad_token_id).to(s2.device)
    #     s2 = torch.cat((s2, p2), dim=0)
    return (s1, s2)


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


def compute_idf(freqs: torch.Tensor) -> torch.Tensor:
    """ compute IDF from word-per-document frequencies """
    # some tokens (pad) appear in every doc, so this works.
    N = freqs.max()
    f = torch.where(freqs == 0, N * 100, freqs)
    idf = (
        ((N - f + 0.5) / (f + 0.5)) + 1
    ).log()
    idf -= idf.min() # set non-existent token IDF to zero
    return idf


def zero_special_tokens_counts(
        tokenizer: transformers.AutoTokenizer,
        counts: torch.Tensor
    ) -> torch.Tensor:

    #######################################
    ### Temporary: don't change counts ###
    # return counts
    #######################################
    
    vocab_size = counts.shape[-1]

    all_special_ids = torch.tensor(tokenizer.all_special_ids)
    special_ids_mask = (
        torch.arange(vocab_size)[:, None] == all_special_ids[None, :]
    ).any(dim=1).to(counts.device)
    zero_counts = torch.zeros_like(counts, device=counts.device)
    return torch.where(special_ids_mask, zero_counts, counts)

def compute_token_counts(
        tokenizer: transformers.AutoTokenizer,
        dataset: datasets.Dataset,
        use_cache: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes histogram, with caching.
    
    Returns three-tuple:
        (i) total number of tokens
        (ii) total documents containing token
        (iii) IDF
    """
    dataset_cache_file = dataset.cache_files[0]['filename']
    cache_file = strip_extension(dataset_cache_file) + '_token_counts.p'

    if not (use_cache and os.path.exists(cache_file)):
        token_freqs, document_token_freqs = compute_token_counts_uncached(
            tokenizer=tokenizer, dataset=dataset
        )
        if use_cache:
            pickle.dump((token_freqs, document_token_freqs), open(cache_file, 'wb'))
    else:
        token_freqs, document_token_freqs = pickle.load(open(cache_file, 'rb'))
    
    # Important: normalize token counts by number of documents.
    token_freqs = token_freqs.float() / len(dataset)
    token_freqs = zero_special_tokens_counts(tokenizer=tokenizer, counts=token_freqs)

    return token_freqs, compute_idf(document_token_freqs)


def get_doc_counts(input_ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Returns the total number occurrences (given a batch of documents in `input_ids`)
    of each given word.

    Returns:
        counts, long torch.Tensor of shape (vocab_size,)
    """
    B = input_ids.shape[0]
    f = torch.zeros((B, vocab_size), dtype=torch.long, device=input_ids.device)
    vals = torch.ones_like(input_ids, device=input_ids.device)
    vocab_totals = f.scatter_add(1, input_ids, vals)
    return vocab_totals.sum(dim=0)


def get_doc_frequencies(input_ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Returns the total number of documents (given a batch of documents in `input_ids`)
    which contain each given word.

    Returns:
        counts, long torch.Tensor of shape (vocab_size,)
    """
    B = input_ids.shape[0]
    f = torch.zeros((B, vocab_size), dtype=torch.long, device=input_ids.device)
    vals = torch.ones_like(input_ids, device=input_ids.device)
    vocab_totals = f.scatter_add(1, input_ids, vals)
    return vocab_totals.clamp(max=1).sum(dim=0)


def get_doc_frequencies_slow(input_ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
    f = torch.zeros((vocab_size,), device=input_ids.device)
    for idx in range(len(input_ids)):
        f += input_ids[idx].bincount(minlength=vocab_size).clamp(max=1)
    return f


def md5_hash(t: Tuple[str]) -> str:
    return hashlib.md5('__'.join(t).encode()).hexdigest()


def md5_hash_kwargs(**kwargs) -> str:
    # We ignore special hf args that start with _ like '__cached__setup_devices'.
    safe_kwargs = {k: str(v) for k,v in kwargs.items() if not k.startswith('_')}
    s = json.dumps(safe_kwargs, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()


def pad_to_same_length(t1: torch.Tensor, t2: torch.Tensor, pad_token: int) -> Tuple[torch.Tensor, torch.Tensor]:
    assert len(t1.shape) == len(t2.shape) == 2
    if t1.shape[1] == t2.shape[1]:
        return (t1, t2)
    elif t1.shape[1] < t2.shape[1]:
        # A is the one we need to pad
        A, B = t1, t2
    else:
        B, A = t1, t2
    
    num_pad_tokens = B.shape[1] - A.shape[1]
    padding = torch.tensor(pad_token, dtype=A.dtype, device=A.device)[None, None]
    padding = padding.repeat((len(A), num_pad_tokens))
    A = torch.cat((A, padding), dim=1)

    if t1.shape[1] < t2.shape[1]:
        # A is the one we need to pad
        return A, B
    else:
        return B, A


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
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def _score(self, query: str, corpus: List[str]) -> float:
        query = self.tokenizer(query, return_tensors='pt').to(device)
        corpus = self.tokenizer(corpus, return_tensors='pt').to(device)
        return self.model(query_embeddings=..., corpus=...)
    
    def rerank(self, 
               corpus: Dict[str, Dict[str, str]], 
               queries: Dict[str, str],
               results: Dict[str, Dict[str, float]],
               top_k: int) -> Dict[str, Dict[str, float]]:
        
        sentence_pairs, pair_ids = [], []
        rerank_scores = []
        
        #### Starting to Rerank using cross-attention
        logging.info("Starting To Rerank Top-{}....".format(top_k))
        
        for query_id in results:
            corpus = []
            if len(results[query_id]) > top_k:
                for (doc_id, _) in sorted(results[query_id].items(), key=lambda item: item[1], reverse=True)[:top_k]:
                    pair_ids.append([query_id, doc_id])
                    corpus_text = (corpus[doc_id].get("title", "") + " " + corpus[doc_id].get("text", "")).strip()
                    corpus.append(corpus_text)
            
            else:
                for doc_id in results[query_id]:
                    pair_ids.append([query_id, doc_id])
                    corpus_text = (corpus[doc_id].get("title", "") + " " + corpus[doc_id].get("text", "")).strip()
                    corpus.append(corpus_text)
            rerank_scores.extend(self._score(queries[query_id], corpus))

        #### Reranking results
        self.rerank_results = {query_id: {} for query_id in results}
        for pair, score in zip(pair_ids, rerank_scores):
            query_id, doc_id = pair[0], pair[1]
            self.rerank_results[query_id][doc_id] = score

        return self.rerank_results 