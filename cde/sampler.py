from typing import Dict, Iterable, List, Optional, Union, Tuple

import abc
import collections
import functools
import logging
import math
import random

import datasets
import numpy as np
import torch
from cde.dataset import FineWebEdu, NomicSupervisedDataset, NomicUnsupervisedDataset
from cde.lib import (
    cluster_dataset,
    cluster_subdomains,
    get_rank, 
    get_world_size, 
    print0,
    shuffle_batches,
    tqdm_if_main_worker,
) 
from cde.lib.cluster_packing import ClusterPackingMixin


NomicDataset = Union[NomicSupervisedDataset, NomicUnsupervisedDataset]


class Sampler(abc.ABC, torch.utils.data.Sampler, ClusterPackingMixin):
    def __init__(
            self, 
            dataset: NomicDataset, 
            batch_size: int, 
            shuffle: bool, 
            max_num_batches: Optional[int] = None,
            num_samples: Optional[int] = None,
            seed: int = 42,
            epoch: int = 0,
            batch_packing_strategy: str = "random",
        ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_num_batches = max_num_batches
        # https://github.com/pytorch/pytorch/blob/main/torch/utils/data/distributed.py#L68
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.num_samples = num_samples or (
            (len(self.dataset) // self.world_size // self.batch_size) 
                * self.batch_size
        )
        # if get_rank() == 0: print0(f"[Sampler] set self.num_samples={self.num_samples} (original num_samples={num_samples})")
        self.total_size = self.num_samples * self.world_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = epoch

    def __hash__(self) -> int:
        return hash(self.__reduce__())

    def __reduce__(self) -> Tuple:
        # this function isn't quite right, but works
        # for caching in streamlit :-)
        return (
            self.dataset._fingerprint, 
            self.batch_size, 
            self.max_num_batches,
            self.total_size,
            self.shuffle,
            self.seed,
            self.epoch,
            get_world_size(),
            self.num_samples,
            self.batch_packing_strategy
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
        chunk_size = (self.total_size // self.world_size)
        my_chunk_start = chunk_size * self.rank
        for i in idxs[my_chunk_start:my_chunk_start+chunk_size]:
            yield i
        
    def __len__(self) -> int:
        logging.debug("Dataloader length: %d // rank: %d // world_size: %d // total: self.total_size", self.num_samples, self.rank, self.world_size, self.total_size)
        return self.num_samples


class FixedSubdomainSampler(RandomSampler):
    """Samples randomly from pre-specified domain during training.
    
    Must have fixed dictionary of subdomains `subdomain_idxs`.
    """
    dataset: NomicDataset
    batch_size: int
    max_num_batches: Optional[int]
    batch_assignments: Dict[int, Iterable[int]]
    batch_packing_strategy: str
    def __init__(
            self, 
            dataset: NomicDataset, 
            batch_size: int, 
            shuffle: bool, 
            share_negatives_between_gpus: bool,
            num_samples: Optional[int] = None,
            batch_packing_strategy = "random",
            seed: int = 42,
        ):
        super().__init__(
            dataset=dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            max_num_batches=None,
            num_samples=num_samples,
            seed=seed,
        )
        self.batch_assignments = self.dataset.subdomain_idxs
        assert hasattr(self.dataset, 'subdomain_idxs')
        self.share_negatives_between_gpus = share_negatives_between_gpus
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        self.batch_packing_strategy = batch_packing_strategy
        if shuffle:
            np_gen = np.random.default_rng(self.seed)
            for k in tqdm_if_main_worker(self.batch_assignments.keys(), desc="Shuffling clusters", colour="red"):
                np_gen.shuffle(self.batch_assignments[k])
    
    def _get_batch_lists(self) -> List[Iterable[int]]:
        """List of sub-batches which will be flattened and reshaped into the full clusters."""
        batch_lists = list(self.batch_assignments.values())
        np_gen = np.random.default_rng(seed=(self.seed + self.epoch))
        batch_lists = self._order_batches(np_gen, batch_lists, cluster_id=0)
        return batch_lists
    
    def _get_indices(self) -> List[int]:
        batch_lists = self._get_batch_lists()
        # 1. Concatenate all datasets from all batches (which should be pre-shuffled once)
        all_assignments = torch.tensor(
            [v for L in tqdm_if_main_worker(batch_lists, desc="Sampler tensorizing clusters") for v in L])
        
        effective_batch_size = (
            self.batch_size * self.world_size 
            if self.share_negatives_between_gpus
            else 
            self.batch_size
        )
        effective_length = len(self.dataset) - (len(self.dataset) % effective_batch_size)
        # 2. Trim off the end (effectively drop_last=True)
        all_assignments = all_assignments[:effective_length]
        num_batches = max(1, int(effective_length // effective_batch_size))
        # 3. Reshape into batches
        all_assignments = all_assignments.reshape(
            (num_batches, effective_batch_size)
        )
        # 4. Randomly reorder batches
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        batch_perm = torch.randperm(num_batches, generator=g)
        # 5. Flatten and return
        print(f"[sampler] finished running get_indices on rank {get_rank()}")
        
        all_indices = shuffle_batches(g, all_assignments[batch_perm])
        return all_indices
        
    def __iter__(self):  
        idxs = self._get_indices()
        num_chunks_per_rank = self.total_size // self.batch_size // self.world_size

        local_chunk_offset = self.batch_size * self.rank
        for chunk_idx in range(num_chunks_per_rank):
            global_chunk_offset = (chunk_idx * self.world_size * self.batch_size)
            chunk_start = global_chunk_offset + local_chunk_offset
            for i in idxs[chunk_start:chunk_start+self.batch_size]:
                yield i


class AutoClusterSampler(FixedSubdomainSampler):
    dataset: Union[NomicDataset, FineWebEdu]
    def __init__(
            self, 
            dataset: NomicDataset,
            query_to_doc: bool, 
            batch_size: int,
            cluster_size: int,
            shuffle: bool,
            share_negatives_between_gpus: bool,
            downscale_and_normalize: bool,
            model: str,
            num_samples: Optional[int] = None,
            batch_packing_strategy: str = "random",
            seed: int = 42,
        ):
        super().__init__(
            dataset=dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_samples=num_samples,
            share_negatives_between_gpus=share_negatives_between_gpus,
            seed=seed,
        )
        self.dataset = dataset
        self.model = model
        self.query_to_doc = query_to_doc
        self.cluster_size = cluster_size
        self.share_negatives_between_gpus = share_negatives_between_gpus
        self.downscale_and_normalize = downscale_and_normalize
        self.batch_packing_strategy = batch_packing_strategy
    
    @property
    def batch_assignments(self) -> Dict[int, List[int]]:
        cluster_assignments = cluster_dataset(
            dataset=self.dataset,
            model=self.model,
            query_key=self.dataset._query_input_ids_key,
            document_key=self.dataset._document_input_ids_key,
            query_to_doc=self.query_to_doc,
            cluster_size=self.cluster_size,
            downscale_and_normalize=self.downscale_and_normalize,
        )
        assert len(self.dataset) == len(cluster_assignments)
        batch_assignments = collections.defaultdict(list)
        for i, cluster in tqdm_if_main_worker(cluster_assignments.items()):
            if isinstance(cluster, list): cluster = cluster[0]
            batch_assignments[cluster].append(i)
        return batch_assignments


class AutoClusterWithinDomainSampler(FixedSubdomainSampler):
    dataset: NomicDataset
    all_cluster_assignments: List[Dict[int, Iterable[int]]]
    def __init__(
            self, 
            dataset: NomicDataset,
            query_to_doc: bool, 
            batch_size: int,
            cluster_size: int,
            shuffle: bool,
            share_negatives_between_gpus: bool,
            downscale_and_normalize: bool,
            model: str,
            num_samples: Optional[int] = None,
            batch_packing_strategy: str = "random",
            seed: int = 42,
        ):
        # TODO: shuffle?
        super().__init__(
            dataset=dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_samples=num_samples,
            share_negatives_between_gpus=share_negatives_between_gpus,
            seed=seed,
        )
        self.batch_assignments = self.dataset.subdomain_idxs
        assert sum(map(len, self.batch_assignments.values())) == len(dataset), f"error: {sum(map(len, self.batch_assignments.values()))} != {len(dataset)}"
        
        self.query_to_doc = query_to_doc
        self.batch_size = batch_size
        self.cluster_size = cluster_size
        self.downscale_and_normalize = downscale_and_normalize
        self.share_negatives_between_gpus = share_negatives_between_gpus
        self.dataset = dataset
        self.model = model
        self.batch_packing_strategy = batch_packing_strategy
    
    @property
    def subdomain_cluster_assignments(self) -> List[Dict[int, List[int]]]:
        return cluster_subdomains(
            dataset=self.dataset,
            subdomains=self.batch_assignments,
            query_to_doc=self.query_to_doc,
            cluster_size=self.cluster_size,
            batch_size=self.batch_size,
            model=self.model,
            downscale_and_normalize=self.downscale_and_normalize,
        )
    
    def _get_batch_lists(self) -> List[Iterable[int]]:
        """Put clusters next to each other in batches."""
        all_batches = []
        np_gen = np.random.default_rng(seed=(self.seed + self.epoch))
        all_cluster_assignments = self.subdomain_cluster_assignments
        for j, minicluster in tqdm_if_main_worker(
                enumerate(all_cluster_assignments), 
                colour="red", 
                leave=False
            ):
            all_sub_batches = []
            for cluster_idx, cluster_list in minicluster.items():
                all_sub_batches.append(cluster_list)
            all_sub_batches = self._order_batches(
                np_gen, 
                all_sub_batches,
                cluster_id=j,
            )
            all_batches.extend(all_sub_batches)
        return all_batches


def get_sampler(
    dataset: datasets.Dataset,
    sampling_strategy: str,
    batch_size: int,
    cluster_size: int,
    shuffle: bool,
    share_negatives_between_gpus: bool,
    clustering_model: str,
    downscale_and_normalize: bool = False,
    clustering_query_to_doc: bool = True,
    num_samples: Optional[int] = None,
    batch_packing_strategy: str = "random",
    seed: int = 42,
) -> Sampler:
    if sampling_strategy == "random":
        return RandomSampler(
            dataset=dataset, 
            batch_size=batch_size,
            shuffle=shuffle,
            num_samples=num_samples,
            seed=seed,
            batch_packing_strategy=batch_packing_strategy,
        )
    elif sampling_strategy == "domain":
        return FixedSubdomainSampler(
            dataset=dataset, 
            batch_size=batch_size,
            shuffle=shuffle,
            share_negatives_between_gpus=share_negatives_between_gpus,
            batch_packing_strategy=batch_packing_strategy,
            num_samples=num_samples,
            seed=seed,
        )
    elif sampling_strategy == "cluster":
        return AutoClusterSampler(
            dataset=dataset, 
            batch_size=batch_size,
            shuffle=shuffle,
            cluster_size=cluster_size,
            downscale_and_normalize=downscale_and_normalize,
            share_negatives_between_gpus=share_negatives_between_gpus,
            batch_packing_strategy=batch_packing_strategy,
            query_to_doc=clustering_query_to_doc, 
            model=clustering_model,
            num_samples=num_samples,
            seed=seed,
        )
    elif sampling_strategy == "cluster_within_domain":
        return AutoClusterWithinDomainSampler(
            dataset=dataset, 
            batch_size=batch_size,
            shuffle=shuffle,
            share_negatives_between_gpus=share_negatives_between_gpus,
            cluster_size=cluster_size,
            batch_packing_strategy=batch_packing_strategy,
            downscale_and_normalize=downscale_and_normalize,
            query_to_doc=clustering_query_to_doc, 
            model=clustering_model,
            num_samples=num_samples,
            seed=seed,
        )
    else:
        raise ValueError(f"unknown sampling strategy {sampling_strategy}")
