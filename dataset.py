from typing import Any, Dict, List, Optional, Tuple

import collections
import copy
import functools
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
import torch.multiprocessing as mp
import tqdm

from lib import (
    datasets_fast_load_from_disk,
    download_url, 
    download_url_and_unzip, 
    embed_with_cache,
    get_tti_cache_dir,
    get_num_proc,
    independent_crop,
    tokenize_dataset, 
)


os.environ['TOKENIZERS_PARALLELISM'] = '0'


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


def get_ance_results(dataset: str, data_path: str, model_name: str = "msmarco-roberta-base-ance-firstp") -> Dict:
    use_parallel = True
    if use_parallel:
        from sentence_transformers import SentenceTransformer
        from mteb import HFDataLoader, RetrievalEvaluator        

        model = SentenceTransformer(model_name)
        model.max_seq_length = 128
        pool = model.start_multi_process_pool()
        def encode(queries, batch_size: int, **kwargs):
            return model.encode_multi_process(queries, batch_size=batch_size, pool=pool)
        model.encode = encode
        print("Loading:", data_path)
        corpus, queries, qrels = HFDataLoader(data_folder=data_path, streaming=False, keep_in_memory=False).load(split="test")
        retriever = RetrievalEvaluator(
            model,
            score_function="dot",
            batch_size=2048,
            corpus_chunk_size=2**18,
        )
        queries = {query['id']: query['text'] for query in queries}
        corpus = {doc['id']: {'title': doc['title'] , 'text': doc['text']} for doc in corpus}
        results = retriever(corpus, queries)
        model.stop_multi_process_pool(pool)
        return results
    else:
        from beir.retrieval import models
        from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
        from beir.datasets.data_loader_hf import HFDataLoader
        from beir.retrieval.evaluation import EvaluateRetrieval
        model = DRES(
            models.SentenceBERT("msmarco-roberta-base-ance-firstp", max_seq_length=128), 
            batch_size=2048, 
            corpus_chunk_size=2**20,
        )
        corpus, queries, qrels = beir.datasets.data_loader.GenericDataLoader(data_path).load(split="test")
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
    ance_results = get_ance_results(dataset=dataset, data_path=data_path)

    corpus = datasets.Dataset.from_list(
        [{"id": k, "text": v["text"]} for k,v in corpus.items()])
    # corpus._fingerprint = md5_hash(f"msmarco_beir_{split}") 

    queries = datasets.Dataset.from_list([{ "id": k, "text": v} for k,v in queries.items()])
    # queries._fingerprint = md5_hash(f"msmarco_beir_{split}") 

    return corpus, queries, qrels, ance_results


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
            self.queries,
            "text",
        )
        self.corpus_embeddings = embed_with_cache(
            embedder, 
            f"{dataset}_corpus" + ("" if split == "train" else f"_{split}"), 
            self.corpus,
            "text",
        )
        self.size = len(self.queries)
    
    def __len__(self) -> int:
        return self.size
    
    def tokenize(self, tokenizer: transformers.PreTrainedTokenizer, max_length: int) -> None:
        os.environ['TOKENIZERS_PARALLELISM'] = '0'
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


class RedditDataset(torch.utils.data.Dataset):
    def __init__(self, data_folder: str, min_examples_per_subreddit: int = 0):
        self.current_dataset_idx: mp.Value = mp.Value('i', 0)
        print(f"Loading Reddit data from path: {data_folder}")
        self.dataset = datasets_fast_load_from_disk(
            os.path.join(
                data_folder, "test.dataset"), 
        )
        print("\tLoading subreddit idxs from disk...")
        self.subreddit_idxs = pickle.load(open(os.path.join(data_folder, "subreddit_idxs.p"), "rb"))
        print("\tLoading subreddit keys from disk...")
        subreddit_names = pickle.load(open(os.path.join(data_folder, "subreddit_keys.p"), "rb"))
        self.subreddit_keys = dict(zip(range(len(subreddit_names)), subreddit_names))
        assert len(self.subreddit_idxs) == len(self.subreddit_keys)

        print(f"Loaded {len(self.dataset)} datapoints from {len(self.subreddit_keys)} subreddits")
        
        self.pad_token_id = 0 # TODO: Set dynamically based on appropriate tokenizer.
        self.dataset.set_format("pt")

        original_num_subreddits = len(self.subreddit_idxs)
        self.min_examples_per_subreddit = min_examples_per_subreddit # TODO: Experiment with this.
        self.subreddit_idxs = { k: v for k,v in self.subreddit_idxs.items() if len(v) > self.min_examples_per_subreddit }
        print(f"Filtered {original_num_subreddits} to {len(self.subreddit_idxs)} with min_examples_per_subreddit={self.min_examples_per_subreddit}")
        self.subdomain_idxs = {}
        self.reset_dataset_idx()
    
    @property
    def fingerprint(self) -> str:
        return self.dataset._fingerprint
    
    def tokenize(self, tokenizer: transformers.PreTrainedTokenizer, max_length: int) -> None:
        # reddit data comes pre-tokenized
        pass

    def __len__(self):
        return len(self.subreddit_keys) * 64

    def reset_dataset_idx(self) -> int:
        dataset_idx = random.choice(list(self.subreddit_idxs.keys()))
        self.current_dataset_idx.value = dataset_idx
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]: 
        # TODO allow other dataset sampling strategies from T0 paper.
        dataset_idx = self.current_dataset_idx.value

        i1 = random.choice(self.subreddit_idxs[dataset_idx])
        i2 = random.choice(self.subreddit_idxs[dataset_idx])

        ex1 = self.dataset[i1]
        ex2 = self.dataset[i2]

        assert ex1["subreddit_idx"] == ex2["subreddit_idx"]

        query_input_ids, document_input_ids = independent_crop(
            ex1["input_ids"],
            pad_token_id=self.pad_token_id,
            l1=256,
            l2=256,
        )

        dataset_input_ids = ex2["input_ids"]
        return {
            'idx': i1,
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


class RedditDatasetWithSupervisedQuestions(RedditDataset):
    def __init__(
            self, data_folder: str,
                 question_folder: str):
        super().__init__(data_folder=data_folder)
        # Load questions
        self.subdomain_idxs = pickle.load(
            open(os.path.join(question_folder, 'question_idxs.p'), 'rb'))
        self.question_dataset = datasets.Dataset.load_from_disk(
            os.path.join(question_folder, 'test.dataset'))
        self.question_dataset.set_format('pt')

        self._embedder_tokenizer_name = 'bert'
        self._dataset_tokenizer_name = 'bert'

        self.size = None

    @property
    def fingerprint(self) -> str:
        return "__".join(
            [self.dataset._fingerprint + self.question_dataset._fingerprint]
        )

    def reset_dataset_idx(self) -> int:
        if not len(self.subdomain_idxs.keys()):
            print("WARNING: Tried to reset dataset w/o any loaded.")
        else:
            dataset_idx = random.choice(list(self.subdomain_idxs.keys()))
            random.shuffle(self.subdomain_idxs[dataset_idx])
            self.current_dataset_idx.value = dataset_idx

    def __len__(self):
        return self.size or sum(map(len, self.subdomain_idxs.values()))
    
    @property
    def _question_input_ids_key(self) -> str:
        """The key in the dataset for question input IDs (tokenizer-specific)."""
        return f'question_input_ids_{self._embedder_tokenizer_name}'
    
    @property
    def _document_input_ids_key(self) -> str:
        """The key in the dataset for document input IDs (tokenizer-specific)."""
        return f'input_ids_{self._embedder_tokenizer_name}'
    
    @property
    def _dataset_input_ids_key(self) -> str:
        """The key in the dataset for dataset input IDs (tokenizer-specific)."""
        return f'input_ids_{self._dataset_tokenizer_name}'
    
    def first(self) -> Dict[str, torch.Tensor]:
        subreddit_question_idxs = list(next(iter(self.subdomain_idxs.values())))
        random_question_idx = subreddit_question_idxs[0]
        return self[random_question_idx]
    
    def __getitem__(self, query_id: int) -> Dict[str, torch.Tensor]: 
        # TODO allow other dataset sampling strategies from T0 paper.
        query_ex = self.question_dataset[query_id]
        query_input_ids = query_ex[self._question_input_ids_key]
        doc_id = query_ex['passage_idx'].item()
        document_input_ids = self.dataset[doc_id][self._document_input_ids_key]

        document_input_ids_dataset_embedder = self.dataset[doc_id][self._dataset_input_ids_key]
        assert query_ex['subreddit_idx'] == self.dataset[doc_id]['subreddit_idx']
        subreddit_idx = query_ex['subreddit_idx'].item()
        random_idx_within_subreddit = random.choice(self.subreddit_idxs[subreddit_idx])
        dataset_input_ids = self.dataset[random_idx_within_subreddit][self._dataset_input_ids_key]
        
        return {
            'idx': doc_id,
            'idx_query': query_id,
            ######################################################################
            'batch_dataset_input_ids': document_input_ids_dataset_embedder,
            'batch_dataset_attention_mask': (document_input_ids_dataset_embedder != self.pad_token_id).int(),
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


class NomicSupervisedDataset:
    num_hard_negatives: int
    tokenizer: transformers.AutoTokenizer
    def __init__(self, tokenizer: transformers.AutoTokenizer, max_seq_length: int, num_hard_negatives: int = 0):
        self.dataset = datasets.load_dataset(
            "nomic-ai/nomic_embed_supervised",
            keep_in_memory=False,
            num_proc=32,
        )["train"]
        self.subdomain_idxs = get_subdomain_idxs_cached(self.dataset)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.num_hard_negatives = num_hard_negatives

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
        
        return {
            'idx': query_id,
            ######################################################################
            "query": ex["query"],
            "document": ex["document"],
            ######################################################################
            # TODO add hard negatives :-)
        }


def get_subdomain_idxs_cached(dataset: datasets.Dataset):
    cache_folder = os.path.join(get_tti_cache_dir(), "subdomain_idxs")
    os.makedirs(cache_folder, exist_ok=True)
    cache_name = dataset._fingerprint + "__sub.p"
    cache_file_path = os.path.join(cache_folder, cache_name)
    if os.path.exists(cache_file_path):
        return pickle.load(open(cache_file_path, 'rb'))
    else:
        subdomain_idxs = collections.defaultdict(list)
        print("Getting subdomains from dataset")
        subdomains = dataset["dataset"]
        for i in tqdm.trange(len(dataset), desc="Counting dataset subdomains"):
            subdomain = subdomains[i]
            subdomain_idxs[subdomain].append(i)
        pickle.dump(subdomain_idxs, open(cache_file_path, 'wb'))
        return subdomain_idxs
        

class NomicUnsupervisedDataset(torch.utils.data.Dataset):
    dataset: datasets.Dataset
    tokenizer: transformers.AutoTokenizer
    def __init__(self, tokenizer: transformers.AutoTokenizer, max_seq_length: int):
        print("[NomicUnsupervisedDataset] loading dataset")
        self.dataset = (
            datasets.load_dataset(
                "nomic-ai/nomic_embed_unsupervised", 
                # keep_in_memory=False,
                # num_proc=32,
            )["train"]
            # datasets_fast_load_from_disk(NOMIC_UNSUPERVISED_DS_PATH)["train"]
        )
        print("[NomicUnsupervisedDataset] loading subdomain idxs")
        self.subdomain_idxs = get_subdomain_idxs_cached(
            dataset=self.dataset
        )
        assert len(self.dataset) == 238_998_494
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
    
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
        return len(self.dataset)
    
    @property
    def _query_input_ids_key(self) -> str:
        """The key in the dataset for question input IDs (tokenizer-specific)."""
        return f'query_input_ids'
    
    @property
    def _document_input_ids_key(self) -> str:
        """The key in the dataset for document input IDs (tokenizer-specific)."""
        return f'document_input_ids'
    
    def __getitem__(self, query_id: int) -> Dict[str, torch.Tensor]: 
        ex = self.dataset[query_id]
        return {
            'idx': query_id,
            ######################################################################
            'query': ex["query"],
            'document': ex["document"],
            ######################################################################
        }


@functools.lru_cache()
def get_char_ids(vocab_size: int, tokenizer_name: str) -> List[torch.Tensor]:
    print("GET_CHAR_IDS")
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
        # reddit data comes pre-tokenized
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


def load_reddit_train_and_val(
        data_folder: str,
        perc: float = 0.9,
        supervised: bool = False
    ) -> Tuple[datasets.Dataset, datasets.Dataset]:
    data_folder_scratch = (
        data_folder.replace(
            "/home/jxm3/research/retrieval/tti3", 
            "/scratch/jxm3/tti3"
        )
    )
    if os.path.exists(data_folder_scratch):
        print(f"Updating reddit data folder from {data_folder} to {data_folder_scratch}; hopefully will be faster")
        data_folder = data_folder_scratch

    question_folder = os.path.join(data_folder, "questions64")
    if supervised:
        train = RedditDatasetWithSupervisedQuestions(
            data_folder=data_folder,
            question_folder=question_folder,
        )
    else:
        train = RedditDataset(data_folder=data_folder)
    print("Initialized dataset:", train.__class__)
    # Randomize subreddit idxs.
    for k in  train.subdomain_idxs.keys():
        random.Random(42).shuffle(train.subdomain_idxs[k])
    # Copy train->val to save dataloading time before split. However need to
    # clone these values individually so that they're not tied together.
    eval = copy.copy(train)
    train.current_dataset_idx: mp.Value = mp.Value('i', 0)
    eval.current_dataset_idx: mp.Value = mp.Value('i', 0)
    subreddit_names = list(train.subreddit_idxs.keys())
    # shuffle with fixed seed so that order doesn't change every time
    random.Random(42).shuffle(subreddit_names)
    N = min(
        1000,
        round(1 - len(subreddit_names) * perc)
    )
    train_subreddits = set(subreddit_names[N:])
    eval_subreddits = set(subreddit_names[:N])

    print(f"Creating training and validation data with a {perc:.2f}/{1-perc:.2f} split ({len(train_subreddits)}/{len(eval_subreddits)})")
    train.subreddit_idxs = { k: v for k,v in train.subreddit_idxs.items() if k in train_subreddits }
    train.subreddit_keys = { k: v for k,v in train.subreddit_keys.items() if k in train_subreddits }
    train.subdomain_idxs= { k: v for k,v in train.subdomain_idxs.items() if k in train_subreddits }
    train.reset_dataset_idx()
    eval.subreddit_idxs = { k: v for k,v in eval.subreddit_idxs.items() if k in eval_subreddits }
    eval.subreddit_keys = { k: v for k,v in eval.subreddit_keys.items() if k in eval_subreddits }
    eval.subdomain_idxs= { k: v for k,v in eval.subdomain_idxs.items() if k in eval_subreddits }
    eval.reset_dataset_idx()

    print("First train point:", train.first())
    return train, eval


def load_synthetic_words_dataset() -> Tuple[datasets.Dataset, datasets.Dataset]:
    train = SyntheticWordsDataset()
    eval = SyntheticWordsDataset()
    eval.max_size = 64 * 8
    return train, eval


if __name__ == '__main__':
    ds_train = NomicSupervisedDataset()
    print(ds_train[0])
