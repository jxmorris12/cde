from typing import Dict, Iterable, List, Optional, Union, Tuple

import abc
import collections
import logging
import math
import random
import torch

import datasets
import torch

from dataset import NomicSupervisedDataset, RedditDataset
from lib import (
    cluster_dataset,
    cluster_subdomains,
    get_rank, 
    get_world_size, 
    tqdm_if_main_worker
) 


class Sampler(abc.ABC, torch.utils.data.Sampler):
    def __init__(
            self, 
            dataset: Union[NomicSupervisedDataset, RedditDataset], 
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

    def __hash__(self) -> int:
        return hash(self.__reduce__())

    def __reduce__(self) -> Tuple:
        # this function isn't quite right, but works
        # for caching in streamlit :-)
        return (
            self.dataset._fingerprint, 
            self.batch_size, 
            self.max_num_batches, 
            self.seed
        )
    
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
    dataset: Union[NomicSupervisedDataset, RedditDataset]
    batch_size: int
    max_num_batches: Optional[int]
    batch_assignments: Dict[int, Iterable[int]]
    def __init__(
            self, 
            dataset: Union[NomicSupervisedDataset, RedditDataset], 
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
        if shuffle:
            for k in tqdm_if_main_worker(self.batch_assignments.keys(), desc="Shuffling clusters", colour="red"):
                random.Random(self.seed).shuffle(self.batch_assignments[k])
        print("running assertion")
        assert sum(map(len, self.batch_assignments.values())) == len(dataset)
        print("done")
    
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
    dataset: Union[NomicSupervisedDataset, RedditDataset]
    def __init__(
            self, 
            dataset: Union[NomicSupervisedDataset, RedditDataset],
            query_to_doc: bool, 
            batch_size: int,
            shuffle: bool,
            model: str,
        ):
        super().__init__(
            dataset=dataset, 
            batch_size=batch_size, 
            shuffle=False,
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


class AutoClusterWithinDomainSampler(FixedSubdomainSampler):
    dataset: Union[NomicSupervisedDataset, RedditDataset]
    def __init__(
            self, 
            dataset: Union[NomicSupervisedDataset, RedditDataset],
            query_to_doc: bool, 
            batch_size: int,
            shuffle: bool,
            model: str,
        ):
        print("calling super().__init__()")
        super().__init__(
            dataset=dataset, 
            batch_size=batch_size, 
            shuffle=False,
        )
        self.dataset = dataset
        self.batch_assignments = cluster_subdomains(
            dataset=self.dataset,
            subdomains=self.batch_assignments,
            query_to_doc=query_to_doc,
            batch_size=batch_size,
            model=model,
        )
        # TODO: shuffle?
        assert sum(map(len, self.batch_assignments.values())) == len(dataset)
        assert len(set(v for b in self.batch_assignments.values() for v in b)) == len(dataset)
        assert set(v for b in self.batch_assignments.values() for v in b) == set(range(len(dataset)))


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
    elif strategy == "cluster_within_domain":
        return AutoClusterWithinDomainSampler(
            dataset=dataset, 
            batch_size=batch_size,
            shuffle=shuffle,
            query_to_doc=data_args.clustering_query_to_doc, 
            model=data_args.clustering_model,
        )
    else:
        raise ValueError(f"invalid sampling strategy {strategy}")
