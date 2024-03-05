from typing import Dict, Iterable, List, Tuple, Union

import collections
import glob
import json
import hashlib
import itertools
import logging
import multiprocessing
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


def get_world_size() -> int:
    try:
        return torch.distributed.get_world_size()
    except (RuntimeError, ValueError):
        return 1


def get_num_proc() -> int:
    world_size: int = get_world_size()
    try:
        # os.sched_getaffinity respects schedulers, unlike cpu_count(), but it's only available
        # on some Unix platforms, so we support both!
        return len(os.sched_getaffinity(0)) // world_size  # type: ignore[attr-defined]
    except AttributeError:
        return multiprocessing.cpu_count() // world_size


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
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
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
        
        pair_ids = []
        rerank_scores_biencoder = []

        query_idx_dict = {key: j for j, key in enumerate(queries['id'])} # Map IDs to idxs
        corpus_idx_dict = {key: j for j, key in enumerate(corpus['id'])} # Map IDs to idxs
        
        #### Starting to Rerank using cross-attention
        logging.info("Starting To Rerank Top-{}....".format(top_k))
        
        for query_id in tqdm.tqdm(results, desc="evaluating dataset"):
            # TODO do this in batch!
            query_text = queries[query_idx_dict[query_id]]["text"]
            minicorpus = []
            minicorpus_text = []
            if len(results[query_id]) > top_k:
                for (doc_id, _) in sorted(results[query_id].items(), key=lambda item: item[1], reverse=True)[:top_k]:
                    pair_ids.append([query_id, doc_id])
                    minicorpus.append(corpus_embeddings[corpus_idx_dict[doc_id]]["embeds"])
                    minicorpus_text.append(corpus[corpus_idx_dict[doc_id]]["text"])
            else:
                for doc_id in results[query_id]:
                    pair_ids.append([query_id, doc_id])
                    minicorpus.append(corpus_embeddings[corpus_idx_dict[doc_id]]["embeds"])
                    minicorpus_text.append(corpus[corpus_idx_dict[doc_id]]["text"])

            query_inputs = self.tokenizer(
                [query_text],
                padding=True,
                truncation=True,
                max_length=self.model.config.max_seq_length,
                return_tensors="pt",
            ).to(device)
            document_inputs = self.tokenizer(
                minicorpus_text,
                padding=True,
                truncation=True,
                max_length=self.model.config.max_seq_length,
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
            
            biencoder_score = torch.nn.functional.cosine_similarity(query_embedding, document_embeddings).flatten().cpu().tolist()
            rerank_scores_biencoder.extend(biencoder_score)

        #### Reranking results
        rerank_results_biencoder = {query_id: {} for query_id in results}
        for pair, biencoder_score in zip(pair_ids, rerank_scores_biencoder):
            query_id, doc_id = pair[0], pair[1]
            rerank_results_biencoder[query_id][doc_id] = biencoder_score

        return rerank_results_biencoder


class ModelConfig(transformers.configuration_utils.PretrainedConfig):
    """We create a dummy configuration class that will just set properties
    based on whatever kwargs we pass in.

    When this class is initialized (see experiments.py) we pass in the
    union of all data, model, and training args, all of which should
    get saved to the config json.
    """

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            try:
                json.dumps(value)
                setattr(self, key, value)
            except TypeError:
                # value was not JSON-serializable, skip
                continue
        super().__init__()


def independent_crop(
    input_ids: torch.Tensor, pad_token_id: int,
    l1: int = 256, l2: int = 256) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns two independent crops from input_ids.
    
    Assumes input_ids has a beginning and end token, like 
        [101, ..., 102, 0, 0, 0].

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

    s1_start = random.randint(1, N-nl1)
    s2_start = random.randint(1, N-nl2)

    s1_idxs = itertools.chain(
        [0], range(s1_start, s1_start+nl1), [N-1]
    )
    s1 = input_ids[torch.tensor(list(s1_idxs))]
    s2_idxs = itertools.chain(
        [0], range(s2_start, s2_start+nl2), [N-1]
    )
    s2 = input_ids[torch.tensor(list(s2_idxs))]
    return (s1, s2)

def load_dataset_tables(
    files: Iterable[str], num_workers: int = 16
) -> Iterable[datasets.table.MemoryMappedTable]:
    import concurrent
    from multiprocessing import Pool

    # num_workers = min(num_workers, len(files))
    num_workers = min(32, len(files))

    use_threads = True
    if use_threads:
        pool_cls = concurrent.futures.ThreadPoolExecutor
        pool_kwargs = {"max_workers": num_workers}
    else:
        pool_cls = Pool
        pool_kwargs = {"processes": num_workers}
    
    with pool_cls(**pool_kwargs) as pool:
        result = list(
            tqdm.tqdm(
                pool.map(datasets.table.MemoryMappedTable.from_file, files),
                desc=f"Loading {len(files)} files with {num_workers} workers",
                total=len(files),
            )
        )
    return result


def datasets_fast_load_from_disk(cache_path: str) -> datasets.Dataset:
    print(f"fast_load_from_disk called with path:", cache_path)
    dataset_info_path = os.path.join(cache_path, "dataset_info.json")
    with open(dataset_info_path, encoding="utf-8") as dataset_info_file:
        dataset_info = datasets.DatasetInfo.from_dict(json.load(dataset_info_file))

    dataset_state_path = os.path.join(cache_path, "state.json")
    with open(dataset_state_path, encoding="utf-8") as state_file:
        state = json.load(state_file)

    files = glob.glob(os.path.join(cache_path, "data-*.arrow"))
    files = sorted(files)
    num_workers = get_num_proc()
    ds_tables = load_dataset_tables(
        files=files,
        num_workers=num_workers
    )
    arrow_table = datasets.table.concat_tables(ds_tables)

    split = state["_split"]
    split = datasets.splits.Split(split) if split is not None else split

    # print("returning dataset")
    return datasets.Dataset(
        arrow_table=arrow_table,
        info=dataset_info,
        split=split,
        fingerprint=state["_fingerprint"],
    )


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


class TensorRunningAverages:
    _store_sum: Dict[str, torch.Tensor]
    _store_total: Dict[str, torch.Tensor]

    def __init__(self):
        self._store_sum = {}
        self._store_total = {}
    
    def __iter__(self) -> Iterable[str]:
        return iter(self._store_sum.keys())

    def update(self, key: str, val: Union[int, float, torch.Tensor]) -> None:
        if key not in self._store_sum:
            self.clear(key)
        if isinstance(val, torch.Tensor):
            val = val.item() # tensor -> num
        self._store_sum[key] += val
        self._store_total[key] += 1

    def get(self, key: str) -> float:
        total = max(self._store_total.get(key).item(), 1.0)
        return (self._store_sum[key] / float(total)).item() or 0.0
    
    def clear(self, key: str) -> None:
        self._store_sum[key] = torch.tensor(0.0, dtype=torch.float32)
        self._store_total[key] = torch.tensor(0, dtype=torch.int32)
    
    def clear_all(self) -> None:
        for key in self._store_sum:
            self.clear(key)

    def get_and_clear_all(self) -> Dict[str, float]:
        metrics = {}
        for key in self:
            metrics[key] = self.get(key)
            self.clear(key)
        return metrics