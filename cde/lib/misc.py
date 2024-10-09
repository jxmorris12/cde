from typing import Dict, Iterable, List, Tuple, Union

import collections
import functools
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
import sys
import zipfile

import datasets
import numpy as np
import safetensors
import torch
import tqdm
import transformers

from cde.lib.dist import get_num_proc, get_rank


def get_cde_cache_dir() -> str:
    script_directory = os.path.normpath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            os.pardir, os.pardir,
        )
    )
    return os.path.join(script_directory, "data")


def get_cache_location_from_kwargs(**kwargs):
    cache_location = os.path.join(
        get_cde_cache_dir(), "cluster"
    )
    os.makedirs(cache_location, exist_ok=True)
    return os.path.join(cache_location, md5_hash_kwargs(**kwargs))


def process_qrels_uncached(corpus: datasets.Dataset, qrels: datasets.Dataset) -> Tuple[Dict[str, List[float]], Dict[str, List[str]]]:
    qrels_idxs = collections.defaultdict(list)
    qrels_scores = collections.defaultdict(list)
    corpus_ids = np.array(corpus['_id'])
    skipped_qrels = 0

    for ex in tqdm.tqdm(qrels, desc='processing qrels', colour='#964B00', leave=False):
        #
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
        logging.warning(f'Warning: Skipped {skipped_qrels}/{len(qrels)} qrels.')
    
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
    print("unzipping =>", zip_file)
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


def tqdm_if_main_worker(iterable: Iterable, **kwargs) -> Iterable:
    if get_rank() == 0:
        return tqdm.tqdm(iterable, **kwargs)
    else:
        return iterable


class ContextualModelConfig(transformers.configuration_utils.PretrainedConfig):
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
        if len(files) > 10:
            files = tqdm_if_main_worker(
                files,
                desc=f"Loading {len(files)} files with {num_workers} workers",
                total=len(files),
                colour="#ffbd88"
            )
        
        result = list(
            pool.map(datasets.table.MemoryMappedTable.from_file, files)
        )
    return result


def datasets_fast_load_from_disk(cache_path: str) -> datasets.Dataset:
    logging.info(f"fast_load_from_disk called with path:", cache_path)
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

def load_embedder_and_tokenizer(name: str) -> Tuple[
        transformers.PreTrainedModel, 
        transformers.PreTrainedTokenizer
]:
    if name.startswith("nomic") or (name == "bert-base-uncased"):
        from cde.lib.nomic_bert import NomicBertModel
        if name.endswith("--from-scratch"):
            name = name.replace("--from-scratch", "")
            config = transformers.AutoConfig.from_pretrained(name, trust_remote_code=True)
            model = NomicBertModel._from_config(config)
        else:
            model = NomicBertModel.from_pretrained(
                name, add_pooling_layer=False
            )
        tokenizer = transformers.AutoTokenizer.from_pretrained(name)
    elif name in ["gtr-base", "gtr_base"]:
        model = transformers.AutoModel.from_pretrained(
            "sentence-transformers/gtr-t5-base"
        ).encoder
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "sentence-transformers/gtr-t5-base"
        )
    elif name == "pile-t5-base-encoder":
        model = transformers.AutoModel.from_pretrained(
            "EleutherAI/pile-t5-base"
        ).encoder
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "EleutherAI/pile-t5-base"
        )
        tokenizer.pad_token = tokenizer.eos_token
    elif name == "pile-t5-base-decoder":
        model = transformers.AutoModel.from_pretrained(
            "EleutherAI/pile-t5-base"
        ).decoder
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "EleutherAI/pile-t5-base"
        )
        tokenizer.pad_token = tokenizer.eos_token
    elif name.startswith("gpt2") or name.startswith("meta-llama") or ("Llama" in name):
        model = transformers.AutoModelForCausalLM.from_pretrained(
            name, 
            # torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True,
            # device_map="auto",
        )
        model.padding_side = "right"
        tokenizer = transformers.AutoTokenizer.from_pretrained(name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_eos_token = True
    else:
        model = transformers.AutoModel.from_pretrained(name, trust_remote_code=True)
        tokenizer = transformers.AutoTokenizer.from_pretrained(name)

        # if use_bettertransformer:
        #     from optimum.bettertransformer import BetterTransformer
        #     model = BetterTransformer.transform(model)
    return model, tokenizer


def inputs_for_key(inputs: Dict[str, torch.Tensor], key: str):
    key += "_"
    return {k.replace(key, ""): v for k,v in inputs.items() if k.startswith(key)}


def count_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0)) 
    except AttributeError:
        return multiprocessing.cpu_count()


def shuffle_batches(g: torch.Generator, list_of_tensors: List[torch.Tensor]) -> List[int]:
    all_indices = []
    for batch_tensor in tqdm_if_main_worker(list_of_tensors, colour="green", desc="Sampler shuffling per-batch"): 
        rand_perm = torch.randperm(len(batch_tensor), generator=g)
        batch_list = batch_tensor[rand_perm].tolist()
        all_indices.extend(batch_list)
    return all_indices


# def shuffle_batches_multiproc(g: torch.Generator, list_of_tensors: List[torch.Tensor], num_processes: int = 8) -> List[int]:
#     all_indices = []
#     print(f"Shuffling {len(list_of_tensors)} tensors with {num_processes} workers.")
#     pbar = tqdm_if_main_worker(list_of_tensors, colour="orange", desc=f"Sampler shuffling per-batch (nproc={num_processes})")
#     pool = multiprocessing.Pool(processes=num_processes) 
#     chunk_size = len(list_of_tensors) // num_processes
#     chunks = [list_of_tensors[i:i + chunk_size] for i in range(0, len(list_of_tensors), chunk_size)]
#     worker_func = functools.partial(shuffle_batches, g=g)
#     results = pool.map(worker_func, chunks)
#     all_indices = []
#     for result in results:
#         all_indices.extend(result)
#         pbar.update()
#     return all_indices


def exit_if_running_or_finished_wandb(
        project_name: str,
        exp_group: str, exp_name: str
    ) -> None:
    print("Checking if experiment is already running...")
    import wandb

    api = wandb.Api()
    running_runs = api.runs(
        path="tti-nomic-7",
        filters={
            "display_name": exp_name,
            "state": {"$regex": "Running|Finished"},
            "config.exp_group": exp_group,
        }  
    )
    print("Found", len(running_runs), f"runs with name {exp_name} and group {exp_group} in {project_name}.")

    if len(running_runs) > 0:
        print("Exiting because experiment is already running or completed.")
        sys.exit(0)
    
