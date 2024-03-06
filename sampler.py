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
from helpers import get_tti_cache_dir, get_rank, get_world_size, md5_hash_kwargs, tqdm_if_main_worker


identity = lambda x: x


def get_cache_location_from_kwargs(**kwargs):
    cache_location = os.path.join(
        get_tti_cache_dir(), "cluster"
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
        dataset_fingerprint=dataset.fingerprint,
        document_key=document_key,
        query_key=query_key,
        batch_size=batch_size,
        model=model,
        query_to_doc=query_to_doc,
    )
    print("checking for cluster at file", clustering_hash)
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
        breakpoint()
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
        self.num_samples = math.ceil(len(self.dataset) / self.world_size)
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
        print("Dataloader length:", self.num_samples, "// rank:", self.rank, "world size:", self.world_size, "total:", self.total_size)
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
        # TODO respect self.shuffle.
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        # 1. Concatenate all datasets from all batches (which should be pre-shuffled once)
        all_assignments = torch.tensor([v for L in self.batch_assignments.values() for v in L])
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
        print("rank", self.rank, "taking idxs", len(idxs), "from", piece_start, "to", piece_start+piece_size)
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
        print(f"got {len(self.batch_assignments)} clusters")


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
    else:
        raise ValueError(f"invalid sampling strategy {strategy}")
