from typing import Iterable, Tuple

import os

import faiss
import torch


from helpers import tqdm_if_main_worker


def paired_kmeans_faiss(
    q: Iterable[torch.Tensor],
    X: Iterable[torch.Tensor], 
    k: int,
    max_iters: int = 100, 
    tol: float = 1e-3, 
    maximize: bool = True,
    initialization_strategy: str = "kmeans++", # ["kmeans++", "random"]
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    paired_vectors = []
    for qi, Xi in zip(q, tqdm_if_main_worker(X, desc='flipping tensors')):
        paired_vectors.append(
            torch.cat((qi, Xi), dim=0)
        )
        paired_vectors.append(
            torch.cat((Xi, qi), dim=0)
        )

    dim = paired_vectors[0].numel()
    kmeans = faiss.Kmeans(
        dim, k, niter=max_iters, gpu=True, verbose=True,
        seed=seed,
    )
    kmeans.train(paired_vectors)

    _distances, assignments = kmeans.index.search(paired_vectors[:len(X)], 1)

    centroids = kmeans.centroids
    breakpoint()

    return centroids, assignments



