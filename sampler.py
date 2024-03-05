from typing import Dict, Iterable, List, Optional, Union

import abc
import collections
import math
import os
import pickle
import random
import torch

import datasets
import tqdm

from cluster_helpers import embed_for_clustering, kmeans, SHOULD_MAXIMIZE_CLUSTER_DISTANCE_FOR_MODEL
from dataset import NomicDataset, RedditDataset
from helpers import md5_hash_kwargs


TTI_CACHE_DIR = os.environ.get(
    "TTI_CACHE_DIR", "/scratch/jxm/tti"
)

identity = lambda x: x


def get_cache_location_from_kwargs(**kwargs):
    cache_location = os.path.join(
        TTI_CACHE_DIR, "cluster"
    )
    os.makedirs(cache_location, exist_ok=True)
    return os.path.join(cache_location, md5_hash_kwargs(**kwargs))


def _cluster_dataset_uncached(
        dataset: datasets.Dataset, 
        query_to_doc: bool,
        model: str,
        query_key: str,
        document_key: str,
        batch_size: int,
    ) -> Dict[int, List[int]]:

    print("processing and tokenizing corpus...")
    max_doc_length = 128
    def ptl(t):
        if len(t) < max_doc_length:
            nz = max_doc_length - len(t) 
            t = torch.cat((t, torch.zeros((nz,))), dim=0)
        return t
    
    document_input_ids = dataset.dataset[document_key]
    document_input_ids = [ptl(t) for t in tqdm.auto.tqdm(document_input_ids)]
    if query_to_doc: 
        query_input_ids = dataset.dataset[query_key]
        query_input_ids = [ptl(t) for t in tqdm.auto.tqdm(document_input_ids)]
    else:
        query_input_ids = document_input_ids
    document_input_ids = torch.stack(document_input_ids)
    query_input_ids = torch.stack(query_input_ids)
    
    q, X = embed_for_clustering(
        query_ids=query_input_ids,
        document_ids=document_input_ids,
        model=model
    )
    
    k = math.ceil(len(X) / batch_size)
    _, assignments = kmeans(
        q=q, 
        X=X,
        k=k,
        maximize=SHOULD_MAXIMIZE_CLUSTER_DISTANCE_FOR_MODEL[model]
    )
    # stack assignments to list.
    assignments_dict = collections.defaultdict(list)
    for i in tqdm.trange(len(assignments), desc="collecting assigments", leave=False):
        assignments_dict[i].append(assignments[i])
    return assignments_dict
    
def cluster_dataset(
        dataset: datasets.Dataset, 
        query_to_doc: bool,
        model: str,
        query_key: str,
        document_key: str,
        batch_size: int,
    ) -> Dict[int, List[int]]:
    # TODO: Turn this caching logic into a nice decorator?
    clustering_hash = get_cache_location_from_kwargs(
        dataset_fingerprint=dataset.fingerprint,
        document_key=document_key,
        query_key=query_key,
        batch_size=batch_size,
        model=model,
        query_to_doc=query_to_doc,
    )
    if os.path.exists(clustering_hash):
        return pickle.load(open(clustering_hash, "rb"))
    else:
        result = _cluster_dataset_uncached(
            dataset=dataset,
            model=model,
            query_to_doc=query_to_doc,
            query_key=query_key,
            document_key=document_key,
            batch_size=batch_size,
        )
        pickle.dump(result, open(clustering_hash, "wb"))
        return result


class Sampler(abc.ABC, torch.utils.data.Sampler):
    def __init__(
            self, 
            dataset: Union[NomicDataset, RedditDataset], 
            batch_size: int, 
            shuffle: bool, 
            max_num_batches: Optional[int] = None,
        ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_num_batches = max_num_batches
        self.shuffle_func = random.shuffle if shuffle else identity
    
    @abc.abstractmethod
    def __iter__(self) -> Iterable[Dict]:
        raise NotImplementedError()
    
    @abc.abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError()


class RandomSampler(Sampler):
    """Samples randomly from a dataset during training."""
    def __iter__(self):
        idxs = list(range(len(self)))
        self.shuffle_func(idxs)
        for i in idxs:
            yield i
        
    def __len__(self) -> int:
        return (
            self.max_num_batches * self.batch_size 
            if self.max_num_batches else len(self.dataset)
        )


class FixedSubdomainSampler(Sampler):
    """Samples randomly from pre-specified domain (subreddits, dataset) during training.
    
    Must have fixed dictionary of subdomains `subdomain_idxs`.
    """
    dataset: Union[NomicDataset, RedditDataset]
    batch_size: int
    max_num_batches: Optional[int]
    batch_assignments: Dict[int, Iterable[int]]
    def __init__(
            self, 
            dataset: Union[NomicDataset, RedditDataset], 
            batch_size: int, 
            shuffle: bool, 
            max_num_batches: Optional[int] = None,
        ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_num_batches = max_num_batches

        len_minus_batch = lambda L: len(L) - (len(L) % batch_size)
        assert hasattr(self.dataset, 'subdomain_idxs')
        num_questions = { k: len_minus_batch(v) for k,v in self.dataset.subdomain_idxs.items() }
        self.batch_assignments = [x for k, v in num_questions.items() for x in [k] * v]
        self.shuffle_func = random.shuffle if shuffle else identity

    def __iter__(self) -> Iterable[Dict]:
        # randomly sample questions per-subreddit (w/o replacement)
        subreddit_idx_pointer = collections.defaultdict(lambda: 0)
        self.shuffle_func(self.batch_assignments)

        for i, subreddit_idx in enumerate(self.batch_assignments):
            # manually track length and stop
            if (i * self.batch_size) >= len(self):
                break

            pointer = subreddit_idx_pointer[subreddit_idx]
            if pointer == 0:
                self.shuffle_func(self.dataset.subdomain_idxs[subreddit_idx])
            subreddit_idx_pointer[subreddit_idx] += 1
            questions = self.dataset.subdomain_idxs[subreddit_idx]

            if (pointer + 1) * self.batch_size > len(questions):
                # drop small batches
                continue
            else:
                for j in range(pointer * self.batch_size, (pointer + 1) * self.batch_size):
                    yield questions[j]

    def __len__(self) -> int:
        if self.max_num_batches:
            return min(len(self.batch_assignments), self.max_num_batches) * self.batch_size
        else:
            return len(self.batch_assignments) * self.batch_size


class AutoClusterSampler(FixedSubdomainSampler):
    dataset: datasets.Dataset
    def __init__(
            self, 
            dataset: datasets.Dataset, 
            query_to_doc: bool, 
            batch_size: int,
            model: str,
        ):
        self.dataset = dataset
        self.batch_assignments = cluster_dataset(
            dataset=dataset,
            model=model,
            query_key=dataset._document_input_ids_key,
            document_key=dataset._query_input_ids_key,
            query_to_doc=query_to_doc,
            batch_size=batch_size,
        )


def get_sampler(
    dataset: datasets.Dataset,
    batch_size: int,
    shuffle: bool,
    data_args,
) -> Sampler:
    strategy = data_args.sampling_strategy
    if strategy == "random":
        return RandomSampler(
            dataset=dataset, 
            batch_size=batch_size,
            shuffle=shuffle,
        )
    elif strategy == "domain":
        return FixedSubdomainSampler(
            dataset=dataset, 
            batch_size=batch_size,
            shuffle=shuffle,
        )
    elif strategy == "cluster":
        return AutoClusterSampler(
            dataset=dataset, 
            batch_size=batch_size,
            query_to_doc=data_args.clustering_query_to_doc, 
            model=data_args.clustering_model,
        )
    else:
        raise ValueError(f"invalid sampling strategy {strategy}")