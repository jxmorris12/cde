from typing import Iterable, Tuple

import os

import faiss
import torch


from helpers import tqdm_if_main_worker


def paired_kmeans_faiss(
    q: torch.Tensor,
    X: torch.Tensor, 
    k: int,
    max_iters: int = 100, 
    tol: float = 1e-3, 
    maximize: bool = True,
    initialization_strategy: str = "kmeans++", # ["kmeans++", "random"]
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    # https://github.com/facebookresearch/faiss/blob/dafdff110489db7587b169a0afee8470f220d295/faiss/python/extra_wrappers.py#L437
    # https://github.com/facebookresearch/faiss/blob/dafdff110489db7587b169a0afee8470f220d295/faiss/Clustering.cpp#L56
    assert q.shape == X.shape
    paired_vectors = torch.cat(
        [
            torch.cat((q, X), dim=0),
            torch.cat((X, q), dim=0),
        ], dim=1
    )
    paired_vectors[0] /= paired_vectors.norm(p=2)

    dim = paired_vectors[0].numel()
    kmeans = faiss.Kmeans(
        dim, k, niter=max_iters, 
        gpu=True, verbose=True,
        spherical=True,
        seed=seed,
    )
    kmeans.train(paired_vectors)

    queries = paired_vectors[:len(q)]
    _distances, assignments = kmeans.index.search(queries, 1)
    assert assignments.shape == (q.shape[0], 1)

    print("Finished kmeans. Average distance:", _distances.mean())

    centroids = torch.tensor(kmeans.centroids)
    assert centroids.shape == (k, paired_vectors.shape[1])
    
    return centroids, assignments



