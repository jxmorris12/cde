from typing import Tuple

import gc

try:
    import faiss
except ImportError as e:
    print("Error loading faiss:", e)
    faiss = None
import torch


def paired_kmeans_faiss(
    q: torch.Tensor,
    X: torch.Tensor, 
    k: int,
    max_iters: int = 100, 
    n_redo: int = 3,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    # https://github.com/facebookresearch/faiss/blob/dafdff110489db7587b169a0afee8470f220d295/faiss/python/extra_wrappers.py#L437
    # https://github.com/facebookresearch/faiss/blob/dafdff110489db7587b169a0afee8470f220d295/faiss/Clustering.cpp#L56
    # https://github.com/facebookresearch/faiss/blob/main/faiss/Clustering.h
    assert q.shape == X.shape
    print("[paired_kmeans_faiss]", q.shape, X.shape, k)
    paired_vectors = torch.cat(
        [
            torch.cat((q, X), dim=0),
            torch.cat((X, q), dim=0),
        ], dim=1
    )
    paired_vectors /= paired_vectors.norm(dim=1, keepdim=True, p=2)
    paired_vectors = paired_vectors.cpu()

    dim = paired_vectors[0].numel()
    # TODO: How to make kmeans use more gpu mem?
    print(f"[paired_kmeans_faiss] initializing Kmeans object (gpu={torch.cuda.is_available()})")
    gc.collect()
    torch.cuda.empty_cache()
    kmeans = faiss.Kmeans(
        dim, k,
        niter=max_iters, 
        nredo=n_redo,
        gpu=torch.cuda.is_available(), 
        verbose=True,
        spherical=True,
        decode_block_size=2**27,
        seed=seed,
    )
    # otherwise the kmeans implementation sub-samples the training set
    # to <= 256 points per centroid
    kmeans.max_points_per_centroid = k * 2
    print("[paired_kmeans_faiss] calling kmeans.train()")
    kmeans.train(paired_vectors)

    queries = paired_vectors[:len(q)]
    _distances, assignments = kmeans.index.search(queries, 1)
    assert assignments.shape == (q.shape[0], 1)

    print("Finished kmeans. Average distance:", _distances.mean())

    centroids = torch.tensor(kmeans.centroids)
    assert centroids.shape == (k, paired_vectors.shape[1])
    
    return centroids, assignments



