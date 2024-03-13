from typing import Dict, Iterable, List, Optional, Union

import abc
import collections
import logging
import math
import os
import pickle
import random
import torch

import datasets
import torch
import tqdm

from cluster_helpers import embed_for_clustering, paired_kmeans, SHOULD_MAXIMIZE_CLUSTER_DISTANCE_FOR_MODEL
from cluster_helpers_faiss import paired_kmeans_faiss
from dataset import NomicDataset, RedditDataset
from helpers import get_tti_cache_dir, get_rank, get_world_size, md5_hash_kwargs, tqdm_if_main_worker


identity = lambda x: x

USE_FAISS = True


def get_cache_location_from_kwargs(**kwargs):
    cache_location = os.path.join(
        get_tti_cache_dir(), "cluster"
    )
    os.makedirs(cache_location, exist_ok=True)
    return os.path.join(cache_location, md5_hash_kwargs(**kwargs))


def _cluster_dataset_uncached(
        dataset: Union[NomicDataset, RedditDataset], 
        query_to_doc: bool,
        model: str,
        query_key: str,
        document_key: str,
        batch_size: int,
    ) -> Dict[int, List[int]]:
    print("Processing and tokenizing corpus for clustering...")
    
    q, X = embed_for_clustering(
        dataset=dataset,
        query_key=query_key,
        document_key=document_key,
        model=model,
        query_to_doc=query_to_doc,
    )
    
    k = math.ceil(len(X) / batch_size / 2)

    if use_faiss:
        _, assignments = paired_kmeans_faiss(
            q=q,
            X=X,
            k=k,
            maximize=SHOULD_MAXIMIZE_CLUSTER_DISTANCE_FOR_MODEL[model]
        )
    else:
        _, assignments = paired_kmeans(
            q=q, 
            X=X,
            k=k,
            maximize=SHOULD_MAXIMIZE_CLUSTER_DISTANCE_FOR_MODEL[model]
        )
    # stack assignments to list.
    assignments_dict = collections.defaultdict(list)
    for i in tqdm.trange(len(assignments), desc="collecting assigments", leave=False):
        assignments_dict[i].append(assignments[i].item())
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
        dataset_fingerprint=dataset._fingerprint,
        document_key=document_key,
        query_key=query_key,
        batch_size=batch_size,
        model=model,
        query_to_doc=query_to_doc,
    )
    print("checking for cluster at file", clustering_hash)
    if os.path.exists(clustering_hash):
        if get_world_size() > 1:
            torch.distributed.barrier()
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
        # https://github.com/pytorch/pytorch/blob/main/torch/utils/data/distributed.py#L68
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.num_samples = math.floor(len(self.dataset) / self.world_size)
        self.total_size = self.num_samples * self.world_size
        self.shuffle = shuffle
        self.seed = 42
        self.epoch = 0
    
    def _get_indices(self) -> Iterable[int]:
        # TODO respect self.shuffle.
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        return torch.randperm(len(self.dataset), generator=g).tolist()

    @abc.abstractmethod
    def __iter__(self) -> Iterable[Dict]:
        raise NotImplementedError()
    
    @abc.abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError()

    def set_epoch(self, epoch: int) -> None:
        r"""
        Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


class RandomSampler(Sampler):
    """Samples randomly from a dataset during training."""    
    def __iter__(self):  
        idxs = self._get_indices()
        for i in idxs[self.rank:self.total_size:self.world_size]:
            yield i
        
    def __len__(self) -> int:
        logging.debug("Dataloader length: %d // rank: %d // world_size: %d // total: self.total_size", self.num_samples, self.rank, self.world_size, self.total_size)
        return self.num_samples


class FixedSubdomainSampler(RandomSampler):
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
        ):
        super().__init__(
            dataset=dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            max_num_batches=None,
        )
        assert hasattr(self.dataset, 'subdomain_idxs')
        self.batch_assignments = self.dataset.subdomain_idxs
        g = torch.Generator()
        g.manual_seed(self.seed)
        for k in self.batch_assignments.keys():
            random.Random(self.seed).shuffle(self.batch_assignments[k])
        assert sum(map(len, self.batch_assignments.values())) == len(dataset)
    
    def _get_indices(self) -> List[int]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        batch_lists = list(self.batch_assignments.values())
        random.Random(self.seed + self.epoch).shuffle(batch_lists)
        # 1. Concatenate all datasets from all batches (which should be pre-shuffled once)
        all_assignments = torch.tensor([v for L in batch_lists for v in L])
        effective_length = len(self.dataset) - (len(self.dataset) % self.batch_size)
        # 2. Trim off the end (effectively drop_last=True)
        all_assignments = all_assignments[:effective_length]
        num_batches = int(effective_length // self.batch_size)
        # 3. Reshape into batches
        all_assignments = all_assignments.reshape(
            (num_batches, self.batch_size)
        )
        # 4. Randomly reorder batches
        batch_perm = torch.randperm(num_batches, generator=g)
        # 5. Flatten and return
        return all_assignments[batch_perm].flatten().tolist()

    def __iter__(self):  
        piece_size = int(math.ceil(self.total_size / self.world_size))
        piece_start = piece_size * self.rank
        idxs = self._get_indices()
        logging.debug("rank", self.rank, "taking idxs", len(idxs), "from", piece_start, "to", piece_start+piece_size)
        for i in idxs[piece_start:piece_start+piece_size]:
            yield i


class AutoClusterSampler(FixedSubdomainSampler):
    dataset: datasets.Dataset
    def __init__(
            self, 
            dataset: datasets.Dataset, 
            query_to_doc: bool, 
            batch_size: int,
            shuffle: bool,
            model: str,
        ):
        super().__init__(
            dataset=dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
        )
        self.dataset = dataset
        cluster_assignments = cluster_dataset(
            dataset=dataset,
            model=model,
            query_key=dataset._document_input_ids_key,
            document_key=dataset._query_input_ids_key,
            query_to_doc=query_to_doc,
            batch_size=batch_size,
        )
        assert len(self.dataset) == len(cluster_assignments)
        self.batch_assignments = collections.defaultdict(list)
        for i, cluster in tqdm_if_main_worker(cluster_assignments.items()):
            if isinstance(cluster, list): cluster = cluster[0]
            self.batch_assignments[cluster].append(i)


class AutoClusterByDomainSampler(FixedSubdomainSampler):
    dataset: datasets.Dataset
    def __init__(
            self, 
            dataset: datasets.Dataset, 
            query_to_doc: bool, 
            batch_size: int,
            shuffle: bool,
            model: str,
        ):
        super().__init__(
            dataset=dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
        )
        self.dataset = dataset
        offset = 0
        final_assignments = collections.defaultdict(list)
        for _, data_idxs in self.batch_assignments.items():
            mini_dataset = dataset.select(data_idxs)
            cluster_assignments = cluster_dataset(
                dataset=mini_dataset,
                model=model,
                query_key=dataset._document_input_ids_key,
                document_key=dataset._query_input_ids_key,
                query_to_doc=query_to_doc,
                batch_size=batch_size,
            )

            for j, cluster in tqdm_if_main_worker(cluster_assignments.items(), leave=False):
                if isinstance(cluster, list): cluster = cluster[0]
                if isinstance(cluster, torch.Tensor): cluster = cluster.item()
                final_assignments[cluster + offset].append(data_idxs[j])
            offset += len(cluster_assignments)
        assert len(self.dataset) == len(cluster_assignments)
        self.batch_assignments = final_assignments
        assert sum(map(len, self.batch_assignments.values())) == len(dataset)


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
            shuffle=shuffle,
            query_to_doc=data_args.clustering_query_to_doc, 
            model=data_args.clustering_model,
        )
    elif strategy == "cluster_in_domain":
        return AutoClusterByDomainSampler(
            dataset=dataset, 
            batch_size=batch_size,
            shuffle=shuffle,
            query_to_doc=data_args.clustering_query_to_doc, 
            model=data_args.clustering_model,
        )
    else:
        raise ValueError(f"invalid sampling strategy {strategy}")
