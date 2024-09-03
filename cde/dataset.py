from typing import Any, Dict, List, Optional, Tuple

import collections
import dataclasses
import functools
import gzip
import json
import logging
import os
import pathlib
import random
import yaml

import datasets
import numpy as np
import pickle
import random
import torch
import transformers
import torch.multiprocessing as mp
import tqdm

from cde.lib import (
    count_cpus,
    datasets_fast_load_from_disk,
    download_url, 
    download_url_and_unzip, 
    get_cde_cache_dir,
    get_num_proc,
    get_rank,
    get_reranking_results,
    get_world_size,
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
        for col in ["query", "document", "negative_document", "text"]:
            if not len(ex.get(col, [])): continue 
            ex_col = ex[col][:max_num_chars]
            tokenized_col = tokenize_fn(ex_col)
            ex[f'{col}_input_ids'] = tokenized_col.input_ids[0]
            ex[f'{col}_attention_mask'] = tokenized_col.attention_mask[0]
        return ex


@dataclasses.dataclass
class GenericTokenizedDataset(TokenizerMixin):
    max_seq_length: int
    tokenizer: transformers.PreTrainedTokenizer
    dataset: datasets.Dataset
    col: str

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.dataset[idx]
        tokenized_output = self._tokenize({
            "document": ex[self.col],
        })
        return {
            "input_ids": tokenized_output["document_input_ids"],
            "attention_mask": tokenized_output["document_attention_mask"],
        }


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
    def __init__(self, tokenizer: transformers.AutoTokenizer, max_seq_length: int, num_hard_negatives: int = 0, use_prefix: bool = False):
        self.dataset = datasets.load_dataset(
            "nomic-ai/nomic_embed_supervised",
            keep_in_memory=False,
            num_proc=32,
        )["train"]
        # self.dataset = self.dataset.select(range(999))
        self.subdomain_idxs = get_subdomain_idxs_cached(self.dataset)
        self.tokenizer = tokenizer
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
        random_idx = random.choice(range(len(self.dataset)))
        random_document = document_prefix + document

        num_hard_negatives = min(self.num_hard_negatives, len(ex["negative"]))
        negative_documents = [document_prefix + d for d in random.sample(ex["negative"], num_hard_negatives)]
        out_ex = self._tokenize({
            "idx": query_id,
            ######################################################################
            "query": query,
            "query_text": query_prefix + ex["query"],
            "query_text__no_prefix": ex["query"],
            "document": document,
            "document_text": document_prefix + ex["document"],
            "document_text__no_prefix": ex["document"],
            ######################################################################
            "random_document": random_document,
            "negative_document": negative_documents, 
            ######################################################################
        })

        ######################################################################
        if "query_embedding" in ex: out_ex["query_embedding"] = torch.Tensor(ex["query_embedding"])
        if "document_embedding" in ex: out_ex["document_embedding"] = torch.Tensor(ex["query_embedding"])
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
        

class NomicUnsupervisedDataset(torch.utils.data.Dataset, TokenizerMixin):
    dataset: datasets.Dataset
    tokenizer: transformers.AutoTokenizer
    max_seq_length: int
    use_prefix: bool
    train_subdomain_key: Optional[str]
    def __init__(self, tokenizer: transformers.AutoTokenizer, max_seq_length: int, use_prefix: bool = False, train_subdomain_key: Optional[str] = None):
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
        self.max_seq_length = max_seq_length

        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        yaml_path = os.path.join(current_file_directory, "config", "nomic_unsupervised.yaml")
        config_yaml = yaml.safe_load(open(yaml_path, "r"))
        self.config = { row["name"]: row for row in config_yaml["datasets"] }
        self.use_prefix = use_prefix
        self.train_subdomain_key = train_subdomain_key

        if self.train_subdomain_key:
            assert self.train_subdomain_key in self.subdomain_idxs, f"can't find key {self.train_subdomain_key} in {self.subdomain_idxs.keys()}"
            self.subdomain_idxs = { self.train_subdomain_key: self.subdomain_idxs[self.train_subdomain_key] }
    
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

            query_prefix = f"{query_prefix}: "
            document_prefix = f"{document_prefix}: "
        else:
            query_prefix = ""
            document_prefix = ""
        query = query_prefix + ex["query"]
        document = document_prefix + ex["document"]
        # random_idx = random.choice(range(len(self.dataset)))
        # print0("__collate__ call [3]")
        out_ex = self._tokenize({
            'idx': idx,
            ######################################################################
            "query": query,
            "query_text": query_prefix + ex["query"],
            "query_text__no_prefix": ex["query"],
            "document": document,
            "document_text": document_prefix + ex["document"],
            "document_text__no_prefix": ex["document"],
            ######################################################################
            # 'random_document': self.dataset[random_idx]["document"],
        })
        if "query_embedding" in ex: out_ex["query_embedding"] = torch.Tensor(ex["query_embedding"])
        if "document_embedding" in ex: out_ex["document_embedding"] = torch.Tensor(ex["query_embedding"])
        return out_ex


@functools.lru_cache()
def get_char_ids(vocab_size: int, tokenizer_name: str) -> List[torch.Tensor]:
    print0("GET_CHAR_IDS")
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    char_strs = [
        f'{(idx+1)}. ' for idx in range(vocab_size)
    ]
    char_strs = [(c * 8).strip() for c in char_strs]
    char_ids = [torch.tensor(t) for t in tokenizer(char_strs).input_ids]
    return char_ids


@functools.lru_cache()
def load_english_single_token_words(
        embedder_tokenizer: transformers.PreTrainedTokenizer, embedder_tokenizer_name: str,
        dataset_tokenizer: transformers.PreTrainedTokenizer, dataset_tokenizer_name: str,
    ) -> datasets.Dataset:
    """Loads all the English words from NLTK dictionary that are a single token
    when a space is appended to them.
    """
    import nltk
    nltk.download('words')
    from nltk.corpus import words
    word_list = words.words()

    def tokenize_ex(ex: Dict[str, str]) -> Tuple[torch.Tensor, torch.Tensor]:
        tt_t5 = embedder_tokenizer(
            [text + " " for text in ex["word"]],
            padding=True, 
            truncation=True,
            max_length=2, 
            add_special_tokens=False,
            return_tensors='pt'
        )
        tt_bert = dataset_tokenizer(
            [text + " " for text in ex["word"]],
            padding=True, 
            truncation=True,
            max_length=2, 
            add_special_tokens=False,
            return_tensors='pt'
        )
        # filter single tokens
        is_single_token_bert = tt_bert.attention_mask.sum(dim=1) == 1
        is_single_token_t5 = tt_t5.attention_mask.sum(dim=1) == 1
        ex["is_single_token"] = is_single_token_bert & is_single_token_t5
        ex[f"input_ids_{embedder_tokenizer_name}"] = tt_t5.input_ids[:, 0].tolist()
        ex[f"input_ids_{dataset_tokenizer_name}"] = tt_bert.input_ids[:, 0].tolist()
        return ex

    word_dataset = datasets.Dataset.from_dict({ "word": word_list })
    word_dataset = word_dataset.map(tokenize_ex, batched=True, batch_size=2000, num_proc=get_num_proc())
    return word_dataset.filter(lambda ex: ex["is_single_token"])


def array_to_categorical(shared_array: mp.Array) -> torch.distributions.Categorical:
    return torch.distributions.Categorical(
        torch.Tensor(
            np.frombuffer(shared_array.get_obj(), dtype=np.float64)
        )
    )


class SyntheticWordsDataset(torch.utils.data.Dataset):
    def __init__(self):
        # TODO pipe in from outside
        self._embedder_tokenizer_name = 't5'
        self._dataset_tokenizer_name = 'bert'
        self.embedder_tokenizer = transformers.AutoTokenizer.from_pretrained('t5-base')
        self.dataset_tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased')

        self.word_list: datasets.Dataset = load_english_single_token_words(
            self.embedder_tokenizer, self._embedder_tokenizer_name,
            self.dataset_tokenizer, self._dataset_tokenizer_name,
        )

        # Create a Zipf distribution
        self.zipf_alpha = 2.0  # Shape parameter, adjust as needed

        self.current_dataset_idx: mp.Value = mp.Value('i', 0)
        self.current_term_frequencies = mp.Array('d', len(self.word_list))
        self.current_term_frequencies_low = mp.Array('d', len(self.word_list))
        self.pad_token_id = 0
        self.reset_dataset_idx()
        self.max_size = None
        self._num_rare_words = 4
        self._num_common_words = 24
    
    def __len__(self) -> int:
        return self.max_size or (len(self.word_list) * 64)

    def tokenize(self, tokenizer: transformers.PreTrainedTokenizer, max_length: int) -> None:
        pass

    def reset_dataset_idx(self) -> int:
        dataset_idx = random.choice(list(range(len(self.word_list))))
        self.current_dataset_idx.value = dataset_idx
        # reset Zipf distribution
        word_counts = np.random.zipf(a=self.zipf_alpha, size=len(self.word_list))
        self.current_term_frequencies[:] = word_counts / word_counts.sum() # set shared array values

        low_word_counts = np.zeros_like(word_counts[:])
        low_freq_words = low_word_counts.argsort()[:len(self.word_list) // 2]
        low_word_counts[low_freq_words] = 1
        self.current_term_frequencies_low[:] = low_word_counts / low_word_counts.sum()
    
    @property
    def _document_input_ids_key(self) -> str:
        """The key in the dataset for document input IDs (tokenizer-specific)."""
        return f'input_ids_{self._embedder_tokenizer_name}'
    
    @property
    def _dataset_input_ids_key(self) -> str:
        """The key in the dataset for dataset input IDs (tokenizer-specific)."""
        return f'input_ids_{self._dataset_tokenizer_name}'
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]: 
        common_terms_dist = array_to_categorical(self.current_term_frequencies)
        rare_terms_dist = array_to_categorical(self.current_term_frequencies_low)

        num_rare_words = np.random.randint(low=1, high=self._num_rare_words)
        num_common_words = 32 - num_rare_words

        # common_terms_1 = common_terms_dist.sample([num_common_words])
        # common_terms_2 = common_terms_dist.sample([num_common_words])
        # common_terms_3 = common_terms_dist.sample([self._num_common_words + self._num_rare_words])

        common_terms_1 = common_terms_dist.probs.argsort(descending=True)[:num_common_words]
        common_terms_2 = common_terms_1.clone()
        common_terms_3 = common_terms_1.clone()

        rare_terms = rare_terms_dist.sample([num_rare_words])

        def doc_ids_to_tensor(ids_tensor: torch.Tensor) -> torch.Tensor:
            ids = ids_tensor.flatten().tolist()
            ids = (
                ([self.embedder_tokenizer.bos_token_id] if (self.embedder_tokenizer.bos_token_id is not None) else []) + 
                [self.word_list[idx][self._document_input_ids_key] for idx in ids] +
                ([self.embedder_tokenizer.eos_token_id] if (self.embedder_tokenizer.eos_token_id is not None) else [])
            )
            return torch.tensor(ids)

        def dataset_ids_to_tensor(ids_tensor: torch.Tensor) -> torch.Tensor:
            ids = ids_tensor.flatten().tolist()
            ids = (
                ([self.dataset_tokenizer.bos_token_id] if (self.dataset_tokenizer.bos_token_id is not None) else []) + 
                [self.word_list[idx][self._dataset_input_ids_key] for idx in ids] +
                ([self.dataset_tokenizer.eos_token_id] if (self.dataset_tokenizer.eos_token_id is not None) else [])
            )
            return torch.tensor(ids)

        Q = torch.cat((common_terms_1, rare_terms), dim=0)
        Q = Q[torch.randperm(Q.shape[0])]
        query_input_ids = doc_ids_to_tensor(Q)

        D = torch.cat((common_terms_2, rare_terms), dim=0)
        D = D[torch.randperm(D.shape[0])]
        document_input_ids = doc_ids_to_tensor(D)

        Z = common_terms_3
        dataset_input_ids = dataset_ids_to_tensor(Z)
        
        return {
            'idx': idx,
            'idx_query': idx,
            ######################################################################
            'dataset_input_ids': dataset_input_ids,
            'dataset_attention_mask': (dataset_input_ids != self.pad_token_id).int(),
            ######################################################################
            'query_input_ids': query_input_ids,
            'query_attention_mask': (query_input_ids != self.pad_token_id).int(),
            ######################################################################
            'document_input_ids': document_input_ids,
            'document_attention_mask': (document_input_ids != self.pad_token_id).int(),
            ######################################################################
        }


def load_synthetic_words_dataset() -> Tuple[datasets.Dataset, datasets.Dataset]:
    train = SyntheticWordsDataset()
    eval = SyntheticWordsDataset()
    eval.max_size = 64 * 8
    return train, eval


if __name__ == '__main__':
    ds_train = NomicSupervisedDataset()
    print0(ds_train[0])
