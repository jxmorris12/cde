from typing import Any, Dict, List, Optional, Tuple

import collections
import functools
import gzip
import json
import logging
import os
import pathlib
import random
import yaml

import datasets
import pickle
import random
import torch
import transformers
import tqdm

from cde.lib import (
    datasets_fast_load_from_disk,
    download_url, 
    get_cde_cache_dir,
    get_rank,
    get_reranking_results,
    print0
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


def load_beir_uncached(dataset: str, split: str, embedder_rerank: str) -> Tuple[datasets.Dataset, datasets.Dataset, Dict[str, Dict[str, int]], Dict]:
    """Loads a BEIR test dataset through tools provided by BeIR.

    Returns:
        corpus (datasets.Dataset): Corpus of documents
            keys -- corpus_id, text
        queries (datasets.Dataset):  Corpus of queries
            keys -- query_id, text
        qrels
        rerank_results
    """
    transformers.logging.set_verbosity_error()
    out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
    data_path = os.path.join(out_dir, dataset)

    print0(
        "[load_beir_uncached] loading dataset uncached", 
        dataset, "... loading split", split
    )
    ### Load BEIR MSMARCO training dataset, this will be used for query and corpus for reference.
    print0("... getting reranking results")
    from mteb import HFDataLoader
    hf_data_loader = HFDataLoader(
        hf_repo=f"mteb/{dataset}", 
        streaming=False, 
        keep_in_memory=False
    )
    corpus, queries, qrels = hf_data_loader.load(split=split)
    rerank_results = get_reranking_results(
        data_path=data_path, 
        split=split,
        model_name=embedder_rerank,
        hf_data_loader=hf_data_loader,
    )
    return corpus, queries, qrels, rerank_results


def load_beir(dataset: str, split: str, embedder_rerank: str) -> Tuple[datasets.Dataset, datasets.Dataset, Dict[str, Dict[str, int]]]:
    if get_rank() == 0:
        print0("[load_beir] loading dataset", dataset)
    cache_path = datasets.config.HF_DATASETS_CACHE # something like /home/jxm3/.cache/huggingface/datasets
    corpus_path = os.path.join(cache_path, f'tti_beir_local_{dataset}_corpus_{split}')
    queries_path = os.path.join(cache_path, f'tti_beir_local_{dataset}_queries_{split}')
    qrels_path = os.path.join(cache_path, f'tti_beir_local_{dataset}_qrels_{split}.p')
    embedder_rerank_path_safe_name = embedder_rerank.replace("/", "__")
    rerank_path = os.path.join(cache_path, f'tti_beir_local_{dataset}_rerank_results_{embedder_rerank_path_safe_name}')

    if os.path.exists(corpus_path) and os.path.exists(queries_path) and os.path.exists(qrels_path) and os.path.exists(rerank_path):
        logging.info(f"Loading {dataset} split %s corpus from path %s", split, corpus_path)
        corpus = datasets_fast_load_from_disk(corpus_path)
        logging.info(f"Loading {dataset} split %s queries from path %s", split, queries_path)
        queries = datasets_fast_load_from_disk(queries_path)
        logging.info(f"Loading {dataset} split %s qrels from path %s", split, qrels_path)
        qrels = pickle.load(open(qrels_path, 'rb'))
        logging.info(f"Loading {dataset} split %s reranking results from path %s", split, rerank_path)
        rerank_results = pickle.load(open(rerank_path, 'rb'))
    else:
        print0("[rerank] will save reranking results to path", rerank_path)
        corpus, queries, qrels, rerank_results = load_beir_uncached(
            dataset=dataset, split=split, embedder_rerank=embedder_rerank,
        )
        logging.info(f"Saving {dataset} split %s corpus to path %s", split, corpus_path)
        corpus.save_to_disk(corpus_path)
        logging.info(f"Saving {dataset} split %s queries to path %s", split, queries_path)
        queries.save_to_disk(queries_path)
        pickle.dump(qrels, open(qrels_path, 'wb'))
        pickle.dump(rerank_results, open(rerank_path, 'wb'))

    return corpus, queries, qrels, rerank_results


class BeirDataset(torch.utils.data.Dataset):
    name: str
    corpus: datasets.Dataset
    queries: datasets.Dataset
    rerank_results: Dict[str, Dict[str, int]]
    size: int
    hard_negatives: Optional[Dict[str, Any]]
    def __init__(
            self,
            dataset: str,
            embedder_rerank: str,
            use_prefix: bool = False,
            split: str = "test"
        ):
        self.name = dataset
        self.use_prefix = use_prefix
        self.corpus, self.queries, self.qrels, self.rerank_results = load_beir(
            dataset=dataset, 
            split=split,
            embedder_rerank=embedder_rerank,
        )
        self.size = len(self.queries)
    
    @property
    def prefix_query(self) -> str:
        return "search_query: " if self.use_prefix else ""
    
    @property
    def prefix_document(self) -> str:
        return "search_document: " if self.use_prefix else ""
    
    def __len__(self) -> int:
        return self.size



class TokenizerMixin:
    def _tokenize(self, ex):
        max_num_chars = 4 * self.max_seq_length
        tokenize_fn = functools.partial(
            self.tokenizer, 
            return_tensors="pt", 
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_length
        )
        if self.first_stage_tokenizer:
            dataset_tokenize_fn = functools.partial(
                self.first_stage_tokenizer, 
                return_tensors="pt", 
                padding="max_length",
                truncation=True,
                max_length=self.max_seq_length
            )
        for col in ["query", "document", "negative_document", "text"]:
            if not len(ex.get(col, [])): continue 
            ex_col = ex[col]
            if isinstance(ex_col, list):
                tokenized_col = tokenize_fn([s[:max_num_chars] for s in ex_col])
                ex[f'{col}_input_ids'] = tokenized_col.input_ids
                ex[f'{col}_attention_mask'] = tokenized_col.attention_mask

                if self.first_stage_tokenizer:
                    tokenized_col = dataset_tokenize_fn([s[:max_num_chars] for s in ex_col])
                    ex[f'{col}_input_ids_first_stage'] = tokenized_col.input_ids
                    ex[f'{col}_attention_mask_first_stage'] = tokenized_col.attention_mask
            elif isinstance(ex_col, str):
                ex_col = ex_col[:max_num_chars]
                if not len(ex_col):
                    ex_col = "[empty]" # Hack to fix empty-string issue from a couple datasets.
                tokenized_col = tokenize_fn(ex_col)
                ex[f'{col}_input_ids'] = tokenized_col.input_ids.squeeze()
                ex[f'{col}_attention_mask'] = tokenized_col.attention_mask.squeeze()

                if self.first_stage_tokenizer:
                    ex_col_no_suffix = ex_col[:max_num_chars]
                    tokenized_col = dataset_tokenize_fn(ex_col_no_suffix)
                    ex[f'{col}_input_ids_first_stage'] = tokenized_col.input_ids.squeeze()
                    ex[f'{col}_attention_mask_first_stage'] = tokenized_col.attention_mask.squeeze()
            else:
                raise ValueError(f"Cannot tokenize value from column {col} of type {type(ex_col)}")
        return ex


class FineWeb(torch.utils.data.Dataset, TokenizerMixin):
    tokenizer: transformers.AutoTokenizer
    max_seq_length: int
    num_hard_negatives: int
    use_prefix: bool
    def __init__(self, tokenizer: transformers.AutoTokenizer, max_seq_length: int, tiny: bool = False):
        self.dataset = datasets.load_dataset(
            "HuggingFaceFW/fineweb", ("sample-10BT" if tiny else "sample-350BT"),
            keep_in_memory=False,
            download_config=datasets.DownloadConfig(resume_download=True),
            num_proc=64,
        )["train"]
        # self.dataset = self.dataset.select(range(999))
        self.subdomain_idxs = { 0: range(len(self.dataset)) }
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        self._query_input_ids_key = "text"
        self._document_input_ids_key = "text"

    def __hash__(self) -> int:
        return hash(self.__reduce__())

    def __reduce__(self) -> Tuple:
        # this function isn't quite right, but works
        # for caching in streamlit :-)
        return (
            self.max_seq_length,
            self._fingerprint,
            self.tokenizer.name_or_path,
        )

    @property
    def _fingerprint(self) -> str:
        return self.dataset._fingerprint

    def reset_dataset_idx(self) -> int:
        pass # Not needed with smart sampler

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]: 
        ex = self.dataset[idx]
        
        return self._tokenize({
            'idx': idx,
            ######################################################################
            "text": ex["text"],
        })


class FineWebEdu(torch.utils.data.Dataset, TokenizerMixin):
    tokenizer: transformers.AutoTokenizer
    max_seq_length: int
    num_hard_negatives: int
    use_prefix: bool
    def __init__(self, tokenizer: transformers.AutoTokenizer, max_seq_length: int, tiny: bool = False):
        self.dataset = datasets.load_dataset(
            "HuggingFaceFW/fineweb", ("sample-10BT" if tiny else "sample-350BT"),
            keep_in_memory=False,
            download_config=datasets.DownloadConfig(resume_download=True),
            num_proc=64,
        )["train"]
        # self.dataset = self.dataset.select(range(999))
        self.subdomain_idxs = { 0: range(len(self.dataset)) }
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        self._query_input_ids_key = "text"
        self._document_input_ids_key = "text"

    def __hash__(self) -> int:
        return hash(self.__reduce__())

    def __reduce__(self) -> Tuple:
        # this function isn't quite right, but works
        # for caching in streamlit :-)
        return (
            self.max_seq_length,
            self._fingerprint,
            self.tokenizer.name_or_path,
        )

    @property
    def _fingerprint(self) -> str:
        return self.dataset._fingerprint

    def reset_dataset_idx(self) -> int:
        pass # Not needed with smart sampler

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]: 
        ex = self.dataset[idx]
        
        return self._tokenize({
            'idx': idx,
            ######################################################################
            "text": ex["text"],
        })

class NomicSupervisedDataset(torch.utils.data.Dataset, TokenizerMixin):
    tokenizer: transformers.AutoTokenizer
    max_seq_length: int
    num_hard_negatives: int
    use_prefix: bool
    def __init__(
            self, 
            tokenizer: transformers.AutoTokenizer, 
            first_stage_tokenizer: Optional[transformers.AutoTokenizer],
            max_seq_length: int,
            num_hard_negatives: int = 0, use_prefix: bool = False
        ):
        self.dataset = datasets.load_dataset(
            "nomic-ai/nomic_embed_supervised",
            keep_in_memory=False,
            num_proc=32,
        )["train"]
        # self.dataset = self.dataset.select(range(999))
        self.subdomain_idxs = get_subdomain_idxs_cached(self.dataset)
        self.tokenizer = tokenizer
        self.first_stage_tokenizer = first_stage_tokenizer
        self.max_seq_length = max_seq_length
        self.num_hard_negatives = num_hard_negatives

        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        yaml_path = os.path.join(current_file_directory, "config", "nomic_supervised.yaml")
        config_yaml = yaml.safe_load(open(yaml_path, "r"))
        self.config = { row["name"]: row for row in config_yaml["datasets"] }
        self.use_prefix = use_prefix
    
    def _remap_dataset_name(self, dataset: str) -> str:
        if dataset == "msmarco_distillation_simlm_rescored_reranked_min15":
            dataset = "msmarco"
        elif dataset == "medi_supernli_sampled":
            dataset = "medi_supernli"
        elif dataset == "medi_sts_wiki_rephrasal":
            dataset = "medi_wiki"
        elif dataset == "medi_sts_stackexchange_dupe":
            dataset = "medi_stackexchange"
        elif dataset == "medi_sts_flickr_sampled":
            dataset = "medi_flickr"
        elif dataset == "fever_hn_mine":
            dataset = "fever"
        elif dataset == "reddit_triples":
            dataset = "reddit"
        elif dataset == "hotpotqa_hn_mine_shuffled":
            dataset = "hotpot"
        elif dataset == "nq_cocondensor_hn_mine_reranked_min15":
            dataset = "nq_triples"
        elif dataset == "nli_simcse_50negs_fixed":
            dataset = "nli_triplets"
        return dataset

    def get_query_prefix(self, dataset: str) -> str:
        dataset = self._remap_dataset_name(dataset)
        return self.config[dataset]["query_prefix"]
    
    def get_document_prefix(self, dataset: str) -> str:
        dataset = self._remap_dataset_name(dataset)
        return self.config[dataset]["document_prefix"]

    def __hash__(self) -> int:
        return hash(self.__reduce__())

    def __reduce__(self) -> Tuple:
        # this function isn't quite right, but works
        # for caching in streamlit :-)
        return (
            self.max_seq_length,
            self._fingerprint,
            self.tokenizer.name_or_path,
        )

    @property
    def _fingerprint(self) -> str:
        return self.dataset._fingerprint

    def reset_dataset_idx(self) -> int:
        pass # Not needed with smart sampler

    def __len__(self):
        return len(self.dataset)
    
    @property
    def _query_input_ids_key(self) -> str:
        """The key in the dataset for question input IDs (tokenizer-specific)."""
        return f'query_input_ids'
    
    @property
    def _document_input_ids_key(self) -> str:
        """The key in the dataset for document input IDs (tokenizer-specific)."""
        return f'document_input_ids'
    
    @property
    def _negative_document_input_ids_key(self) -> str:
        """The key in the dataset for document input IDs (tokenizer-specific)."""
        return f'hn_input_ids'
    
    @property
    def _dataset_input_ids_key(self) -> str:
        """The key in the dataset for dataset input IDs (tokenizer-specific)."""
        return f'document_input_ids'
    
    def first(self) -> Dict[str, torch.Tensor]:
        subdomain_query_idxs = list(next(iter(self.subdomain_idxs.values())))
        random_query_idx = subdomain_query_idxs[0]
        return self[random_query_idx]
    
    def __getitem__(self, query_id: int) -> Dict[str, torch.Tensor]: 
        ex = self.dataset[query_id]
        # print0("NomicSupervisedDataset.__getitem__", query_id)
        
        if self.use_prefix:
            query_prefix = self.get_query_prefix(ex["dataset"])
            document_prefix = self.get_document_prefix(ex["dataset"])
            query_prefix = f"{query_prefix}: "
            document_prefix = f"{document_prefix}: "
        else:
            query_prefix = ""
            document_prefix = ""
        query = query_prefix + ex["query"]
        document = document_prefix + ex["document"]
        random_document = document_prefix + document

        num_hard_negatives = min(self.num_hard_negatives, len(ex["negative"]))
        negative_documents_raw_text = random.sample(ex["negative"], num_hard_negatives)
        negative_documents = [document_prefix + d for d in negative_documents_raw_text]
        out_ex = self._tokenize({
            "idx": query_id,
            ######################################################################
            "query": query,
            "query_prefix": query_prefix,
            "query_text": query_prefix + ex["query"],
            "query_text__no_prefix": ex["query"],
            "document_prefix": document_prefix,
            "document": document,
            "document_text": document_prefix + ex["document"],
            "document_text__no_prefix": ex["document"],
            ######################################################################
            "random_document": random_document,
            "negative_document": negative_documents, 
            "negative_document_text": negative_documents,
            "negative_document_text__no_prefix": negative_documents_raw_text,
            "negative_document_idx": [query_id] * len(negative_documents),
            ######################################################################
        })
        ######################################################################
        if "query_embedding" in ex: out_ex["query_embedding"] = ex["query_embedding"]
        if "document_embedding" in ex: out_ex["document_embedding"] = ex["query_embedding"]
        return out_ex
    

class BGEDataset(torch.utils.data.Dataset, TokenizerMixin):
    def __init__(
            self, 
            tokenizer: transformers.AutoTokenizer, 
            first_stage_tokenizer: Optional[transformers.AutoTokenizer], 
            max_seq_length: int, 
            num_hard_negatives: int = 0, 
            use_prefix: bool = False,
            use_short_prefix: bool = True,
        ):
        dataset = datasets.load_dataset(
            "cfli/bge-full-data",
            num_proc=32,
        )
        self.dataset = process_bge_dataset_cached(dataset)
        self.subdomain_idxs = get_subdomain_idxs_cached(dataset=self.dataset)
        self.tokenizer = tokenizer
        self.first_stage_tokenizer = first_stage_tokenizer
        self.max_seq_length = max_seq_length
        self.num_hard_negatives = num_hard_negatives
        self.use_prefix = use_prefix
        self.use_short_prefix = use_short_prefix
        
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        yaml_path = os.path.join(current_file_directory, "config", "bge.yaml")
        config_yaml = yaml.safe_load(open(yaml_path, "r"))
        self.config = { row["name"]: row for row in config_yaml["datasets"] }
        self.use_prefix = use_prefix

    def get_query_prefix(self, dataset: str) -> str:
        return self.config[dataset]["query_prefix"]
    
    def get_document_prefix(self, dataset: str) -> str:
        return self.config[dataset]["document_prefix"]

    def __hash__(self) -> int:
        return hash(self.__reduce__())

    def __reduce__(self) -> Tuple:
        # this function isn't quite right, but works
        # for caching in streamlit :-)
        return (
            self.max_seq_length,
            self._fingerprint,
            self.tokenizer.name_or_path,
        )

    @property
    def _fingerprint(self) -> str:
        return self.dataset._fingerprint

    def reset_dataset_idx(self) -> int:
        pass # Not needed with smart sampler

    def __len__(self):
        return len(self.dataset)
    
    @property
    def _query_input_ids_key(self) -> str:
        """The key in the dataset for question input IDs (tokenizer-specific)."""
        return f'query_input_ids'
    
    @property
    def _document_input_ids_key(self) -> str:
        """The key in the dataset for document input IDs (tokenizer-specific)."""
        return f'document_input_ids'
    
    @property
    def _negative_document_input_ids_key(self) -> str:
        """The key in the dataset for document input IDs (tokenizer-specific)."""
        return f'hn_input_ids'
    
    @property
    def _dataset_input_ids_key(self) -> str:
        """The key in the dataset for dataset input IDs (tokenizer-specific)."""
        return f'document_input_ids'
    
    def first(self) -> Dict[str, torch.Tensor]:
        subdomain_query_idxs = list(next(iter(self.subdomain_idxs.values())))
        random_query_idx = subdomain_query_idxs[0]
        return self[random_query_idx]
    
    def __getitem__(self, query_id: int) -> Dict[str, torch.Tensor]: 
        ex = self.dataset[query_id]

        short_query_prefix = self.get_query_prefix(ex["dataset"])
        short_document_prefix = self.get_document_prefix(ex["dataset"])
        is_symmetric = (short_query_prefix == short_document_prefix)

        if self.use_prefix:
            if self.use_short_prefix:
                query_prefix = f"{short_query_prefix}: "
                document_prefix = f"{short_document_prefix}: "
            else:
                query_prefix = ex["prompt"] + self.tokenizer.bos_token + " "
                document_prefix = query_prefix if is_symmetric else ""
        else:
            query_prefix = ""
            document_prefix = ""
        query = query_prefix + ex["query"]
        document = document_prefix + ex["document"]
        random_document = document_prefix + document

        if len(document) == 0:
            # fix for empty documents
            document = "[empty]"

        # print(ex["dataset"], is_symmetric, "/", query_prefix, "/", document_prefix)

        if len(ex["neg"]) < self.num_hard_negatives:
            # sample w/ replacement
            negative_documents_raw_text = random.choices(ex["neg"], k=self.num_hard_negatives)
        else:
            # sample w/o replacement
            negative_documents_raw_text = random.sample(ex["neg"], self.num_hard_negatives)
        negative_documents = [document_prefix + d for d in negative_documents_raw_text]
        out_ex = self._tokenize({
            "idx": query_id,
            ######################################################################
            "query": query,
            "query_prefix": query_prefix,
            "query_text": query_prefix + ex["query"],
            "query_text__no_prefix": ex["query"],
            "document_prefix": document_prefix,
            "document": document,
            "document_text": document_prefix + document,
            "document_text__no_prefix": document,
            ######################################################################
            "random_document": random_document,
            "negative_document": negative_documents, 
            "negative_document_text": negative_documents,
            "negative_document_text__no_prefix": negative_documents_raw_text,
            "negative_document_idx": [query_id] * len(negative_documents),
            ######################################################################
        })

        ######################################################################
        if "query_embedding" in ex: out_ex["query_embedding"] = ex["query_embedding"]
        if "document_embedding" in ex: out_ex["document_embedding"] = ex["query_embedding"]
        return out_ex


def get_subdomain_idxs_cached(dataset: datasets.Dataset):
    cache_folder = os.path.join(get_cde_cache_dir(), "subdomain_idxs")
    os.makedirs(cache_folder, exist_ok=True)
    cache_name = dataset._fingerprint + "__sub.p"
    cache_file_path = os.path.join(cache_folder, cache_name)
    if os.path.exists(cache_file_path):
        return pickle.load(open(cache_file_path, 'rb'))
    else:
        subdomain_idxs = collections.defaultdict(list)
        print0("Getting subdomains from dataset")
        subdomains = dataset["dataset"]
        for i in tqdm.trange(len(dataset), desc="Counting dataset subdomains", colour="blue"):
            subdomain = subdomains[i]
            subdomain_idxs[subdomain].append(i)
        pickle.dump(subdomain_idxs, open(cache_file_path, 'wb'))
        return subdomain_idxs


def _flatten_docs(ex):
    ex["document"] = ex["document"][0]
    return ex


def process_bge_dataset_cached(dataset: datasets.Dataset):
    cache_folder = os.path.join(get_cde_cache_dir(), "bge")
    os.makedirs(cache_folder, exist_ok=True)
    cache_name = "bge_processed.dataset"
    cache_file_path = os.path.join(cache_folder, cache_name)
    if os.path.exists(cache_file_path):
        return datasets_fast_load_from_disk(cache_file_path)
    else:
        splits = []
        for split_name in sorted(dataset.keys()):
            split = dataset[split_name]
            split = split.add_column("dataset", [split_name] * len(split))
            split = split.rename_column("pos", "document")
            split = split.map(_flatten_docs)
            splits.append(split)
        dataset = datasets.concatenate_datasets(splits)
        dataset.save_to_disk(cache_file_path, num_proc=32)
        return dataset
        

class NomicUnsupervisedDataset(torch.utils.data.Dataset, TokenizerMixin):
    dataset: datasets.Dataset
    tokenizer: transformers.AutoTokenizer
    max_seq_length: int
    use_prefix: bool
    train_subdomain_key: Optional[str]
    def __init__(self, 
            tokenizer: transformers.AutoTokenizer, 
            first_stage_tokenizer: Optional[transformers.AutoTokenizer],
            max_seq_length: int, 
            use_prefix: bool = False, 
            use_short_prefix: bool = False,
            train_subdomain_key: Optional[str] = None
        ):
        print0("[NomicUnsupervisedDataset] loading dataset")
        self.dataset = (
            datasets.load_dataset(
                "nomic-ai/nomic_embed_unsupervised",
                num_proc=32,
                keep_in_memory=False,
            )["train"]
        )
        print0("[NomicUnsupervisedDataset] loading subdomain idxs")
        # TODO: Share dict between processes.
        self.subdomain_idxs = get_subdomain_idxs_cached(dataset=self.dataset)
        assert len(self.dataset) == 238_998_494
        self.tokenizer = tokenizer
        self.first_stage_tokenizer = first_stage_tokenizer
        self.max_seq_length = max_seq_length

        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        self.use_short_prefix = use_short_prefix
        if use_short_prefix:
            yaml_path = os.path.join(current_file_directory, "config", "nomic_unsupervised.yaml")
        else:
            yaml_path =  os.path.join(current_file_directory, "config", "nomic_unsupervised_long_prefix.yaml")
        config_yaml = yaml.safe_load(open(yaml_path, "r"))
        self.config = { row["name"]: row for row in config_yaml["datasets"] }
        self.use_prefix = use_prefix
        self.train_subdomain_key = train_subdomain_key
        if self.train_subdomain_key:
            assert self.train_subdomain_key in self.subdomain_idxs, f"can't find key {self.train_subdomain_key} in {self.subdomain_idxs.keys()}"
            self.subdomain_idxs = { self.train_subdomain_key: self.subdomain_idxs[self.train_subdomain_key] }
        
    
    def get_query_prefix(self, dataset: str) -> str:
        if self.use_short_prefix:
            return self.config[dataset]["query_prefix"] + ": "
        else:
            return self.config[dataset]["query_prefix"] + self.tokenizer.bos_token + " "
    
    def get_document_prefix(self, dataset: str) -> str:
        if self.use_short_prefix:
            return self.config[dataset]["document_prefix"] + ": "
        else:
            # For long prefixes, we either duplicate the query prefix ("query-only")
            # or just don't use one at all (most non-STS datasets).
            if self.config[dataset].get("is_symmetric", False):
                return self.get_query_prefix(dataset)
            else:
                return ""
    
    def __hash__(self) -> int:
        return hash(self.__reduce__())

    def __reduce__(self) -> Tuple:
        # this function isn't quite right, but works
        # for caching in streamlit :-)
        return (
            self._fingerprint, 
            self.max_seq_length,
            self.tokenizer.name_or_path,
        )
    @property
    def _fingerprint(self) -> str:
        return self.dataset._fingerprint

    def reset_dataset_idx(self) -> int:
        pass # Not needed with smart sampler

    def __len__(self):
        if self.train_subdomain_key:
            # Support training on a single subdomain.
            return len(self.subdomain_idxs[self.train_subdomain_key])
        else:
            return len(self.dataset)
    
    @property
    def _query_input_ids_key(self) -> str:
        """The key in the dataset for question input IDs (tokenizer-specific)."""
        return f'query_input_ids'
    
    @property
    def _document_input_ids_key(self) -> str:
        """The key in the dataset for document input IDs (tokenizer-specific)."""
        return f'document_input_ids'
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]: 
        ex = self.dataset[idx]

        if self.use_prefix:
            query_prefix = self.get_query_prefix(ex["dataset"])
            document_prefix = self.get_document_prefix(ex["dataset"])
        else:
            query_prefix = ""
            document_prefix = ""
        query = query_prefix + ex["query"]
        document = document_prefix + ex["document"]
        # random_idx = random.choice(range(len(self.dataset)))
        # print0("__getitem__ call [3]")
        # print0("__getitem__ =>", ex.keys())
        out_ex = self._tokenize({
            'idx': idx,
            ######################################################################
            "query": query,
            "query_text": query_prefix + ex["query"],
            "query_text__no_prefix": ex["query"],
            "query_prefix": query_prefix,
            "document": document,
            "document_text": document_prefix + ex["document"],
            "document_text__no_prefix": ex["document"],
            "document_prefix": document_prefix,
            ######################################################################
            # 'random_document': self.dataset[random_idx]["document"],
        })
        if "query_embedding" in ex: out_ex["query_embedding"] = torch.Tensor(ex["query_embedding"])
        if "document_embedding" in ex: out_ex["document_embedding"] = torch.Tensor(ex["query_embedding"])
        return out_ex


if __name__ == '__main__':
    ds_train = NomicSupervisedDataset()
    print0(ds_train[0])
