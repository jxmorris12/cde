from typing import Any, Dict, Iterable, List, Optional, Tuple

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
from helpers import (
    datasets_fast_load_from_disk,
    download_url, download_url_and_unzip, 
    get_num_proc,
    independent_crop,
    tokenize_dataset, 
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
    embedder_cache_path = embedder.replace('/', '__')
    # cache_folder = datasets.config.HF_DATASETS_CACHE
    cache_folder = "/scratch/jxm3"
    cache_folder = os.path.join(cache_folder, 'corpus_embeddings', embedder_cache_path)
    os.makedirs(cache_folder, exist_ok=True)
    cache_path = os.path.join(cache_folder, cache_name) #  + "_small")


    if os.path.exists(cache_path):
        print("[embed_with_cache] Loading embeddings at path:", cache_path)
        return datasets_fast_load_from_disk(cache_path)

    print("[embed_with_cache] computing embeddings to save at path:", cache_path)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(embedder)
    model.max_seq_length = 512
    embeddings = model.encode(texts, batch_size=64,  show_progress_bar=True)

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
    def __init__(self, data_folder: str, min_examples_per_subreddit: int = 256):
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

        # for key in self.subreddit_idxs.keys():
            # random.shuffle(self.subreddit_idxs[key])
        
        self.pad_token_id = 0 # TODO: Set dynamically based on appropriate tokenizer.
        self.dataset.set_format("pt")

        original_num_subreddits = len(self.subreddit_idxs)
        self.min_examples_per_subreddit = min_examples_per_subreddit # TODO: Experiment with this.
        self.subreddit_idxs = { k: v for k,v in self.subreddit_idxs.items() if len(v) > self.min_examples_per_subreddit }
        print(f"Filtered {original_num_subreddits} to {len(self.subreddit_idxs)} with min_examples_per_subreddit={self.min_examples_per_subreddit}")
        self.subreddit_questions = {}
        self.reset_dataset_idx()
    
    def tokenize(self, tokenizer: transformers.PreTrainedTokenizer, max_length: int) -> None:
        # reddit data comes pre-tokenized
        pass

    def __len__(self):
        return len(self.subreddit_keys)

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
        self.subreddit_questions = pickle.load(
            open(os.path.join(question_folder, 'question_idxs.p'), 'rb'))
        self.question_dataset = datasets.Dataset.load_from_disk(
            os.path.join(question_folder, 'test.dataset'))
        self.question_dataset.set_format('pt')

        self._embedder_tokenizer_name = 't5'
        self._dataset_tokenizer_name = 'bert'

    def reset_dataset_idx(self) -> int:
        if not len(self.subreddit_questions.keys()):
            print("WARNING: Tried to reset dataset w/o any loaded.")
        else:
            dataset_idx = random.choice(list(self.subreddit_questions.keys()))
            self.current_dataset_idx.value = dataset_idx

    def __len__(self):
        return len(self.subreddit_questions) * 64
    
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
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]: 
        # TODO allow other dataset sampling strategies from T0 paper.
        dataset_idx = self.current_dataset_idx.value        

        dataset_questions = self.subreddit_questions[dataset_idx]
        query_id = dataset_questions[idx % len(dataset_questions)]
        query_ex = self.question_dataset[query_id]
        query_input_ids = query_ex[self._question_input_ids_key]
        doc_id = query_ex['passage_idx'].item()
        document_input_ids = self.dataset[doc_id][self._document_input_ids_key]

        assert query_ex['subreddit_idx'] == self.dataset[doc_id]['subreddit_idx']

        subreddit_idx = query_ex['subreddit_idx'].item()
        random_idx_within_subreddit = random.choice(self.subreddit_idxs[subreddit_idx])
        dataset_input_ids = self.dataset[random_idx_within_subreddit][self._dataset_input_ids_key]
        
        return {
            'idx': doc_id,
            'idx_query': query_id,
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
    ) -> datasets.Dataset:
    """Loads all the English words from NLTK dictionary that are a single token
    when a space is appended to them.
    """
    import nltk
    nltk.download('words')
    from nltk.corpus import words
    word_list = words.words()
    t5_tokenizer = transformers.AutoTokenizer.from_pretrained('t5-base')
    bert_tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased')

    def tokenize_ex(ex: Dict[str, str]) -> Tuple[torch.Tensor, torch.Tensor]:
        tt_bert = bert_tokenizer(
            [text + " " for text in ex["word"]],
            padding=True, 
            truncation=True,
            max_length=4, 
            add_special_tokens=False,
            return_tensors='pt'
        )
        tt_t5 = t5_tokenizer(
            [text + " " for text in ex["word"]],
            padding=True, 
            truncation=True,
            max_length=4, 
            add_special_tokens=False,
            return_tensors='pt'
        )
        # filter single tokens
        ex["input_ids_t5"] = tt_t5.input_ids
        ex["input_ids_bert"] = tt_bert.input_ids
        is_single_token_bert = tt_bert.attention_mask.sum(dim=1) == 1
        is_single_token_t5 = tt_t5.attention_mask.sum(dim=1) == 1
        ex["is_single_token"] = is_single_token_bert & is_single_token_t5
        return ex

    word_dataset = datasets.Dataset.from_dict({ "word": word_list })
    word_dataset = word_dataset.map(tokenize_ex, batched=True, batch_size=2000, num_proc=get_num_proc())
    return word_dataset.filter(lambda ex: ex["is_single_token"])

class SyntheticWordsDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.word_list = ['j', 'a', 'c', 'k'] # load_english_single_token_words()
        num_samples = 1000

        # Create a Zipf distribution
        alpha = 2.0  # Shape parameter, adjust as needed

        self.current_dataset_idx: mp.Value = mp.Value('i', 0)
        self.current_term_frequencies = mp.Array('d', len(self.word_list))
        self.pad_token_id = 0
        self.reset_dataset_idx()
        self.max_size = None
    
    def __len__(self) -> int:
        return self.max_size or (len(self.word_list) * 64)

    def tokenize(self, tokenizer: transformers.PreTrainedTokenizer, max_length: int) -> None:
        # reddit data comes pre-tokenized
        pass

    def reset_dataset_idx(self) -> int:
        dataset_idx = random.choice(list(range(self.word_list)))
        self.current_dataset_idx.value = dataset_idx
        # reset Zipf distribution
        samples = np.random.zipf(a=4.0, n=len(self.word_list))
        self.current_term_frequencies[:] = samples[:] # set shared array values
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]: 
        doc_id = random.randint(0, self.vocab_size - 1)
        document_input_ids = self.char_ids[doc_id]

        query_id = (doc_id + self.current_dataset_idx.value) % self.vocab_size
        query_input_ids = self.char_ids[query_id]

        # Random mappings
        ex_id = random.randint(0, self.vocab_size - 1)
        ex_id_2 = (ex_id + self.current_dataset_idx.value) % self.vocab_size
        dataset_input_ids = torch.cat(
            (
                self.char_ids[ex_id][:-1], # cut off EOS
                self.char_ids[ex_id_2][1:] # cut off BOS :)
            ), dim=0
        )
        
        return {
            'idx': doc_id,
            'idx_query': query_id,
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
                             ) -> Tuple[
    torch.utils.data.Dataset, torch.utils.data.Dataset]:

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
    for k in  train.subreddit_questions.keys():
        random.Random(42).shuffle(train.subreddit_questions[k])
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
    train.subreddit_questions = { k: v for k,v in train.subreddit_questions.items() if k in train_subreddits }
    train.reset_dataset_idx()
    eval.subreddit_idxs = { k: v for k,v in eval.subreddit_idxs.items() if k in eval_subreddits }
    eval.subreddit_keys = { k: v for k,v in eval.subreddit_keys.items() if k in eval_subreddits }
    eval.subreddit_questions = { k: v for k,v in eval.subreddit_questions.items() if k in eval_subreddits }
    eval.reset_dataset_idx()

    print("First train point:", train[0])
    return train, eval


def load_synthetic_words_dataset():
    train = SyntheticWordsDataset()
    eval = SyntheticWordsDataset()
    return train, eval


if __name__ == '__main__':
    # nfcorpus = BeirDataset(
    #     dataset="nfcorpus",    
    #     embedder="sentence-transformers/gtr-t5-base"
    # )
    # dataset = MsmarcoDatasetHardNegatives(
    #     embedder="sentence-transformers/gtr-t5-base",
    #     num_hard_negatives=1,
    # )
    # print(dataset[10_001])

    # ds_train, ds_val = load_reddit_train_and_val(
    #     data_folder="/home/jxm3/research/retrieval/tti3/data/full",
    #     perc=0.9,
    # )
    ds_train, ds_val = load_synthetic_words_dataset()
    print(ds_train[0])
