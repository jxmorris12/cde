from typing import Callable, Tuple

import math
import psutil

import scipy
import torch
import tqdm

from bm25_pt.bm25 import TokenizedBM25


# Whether to maximize distances
# between points during clustering.
SHOULD_MAXIMIZE_CLUSTER_DISTANCE_FOR_MODEL = {
    "bm25": True,
    "nomic_embed": False,
}

DEBUG_MEM = False


def embed_for_clustering(
        query_ids: torch.Tensor,
        document_ids: torch.Tensor,
        model: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    if model == "bm25":
        vocab_size = int(document_ids.max() + 1)
        bm25 = TokenizedBM25(vocab_size=vocab_size)
        bm25.index(document_ids)
        queries = bm25.docs_to_bags(query_ids).to_sparse_coo()
        corpus = bm25._corpus_scores.to_sparse_coo()
        return queries, corpus
    elif model == "nomic_embed":
        # TODO compute embeddings here
        raise NotImplementedError()
    else:
        raise ValueError(f"model {model} not supported")


def slice_sparse_tensor_rows(t: torch.sparse.Tensor, min_row: int, max_row: int) -> torch.sparse.Tensor:
    t = t.coalesce()
    row_idxs = t.indices()[0]
    index_mask = (min_row <= row_idxs) & (row_idxs < max_row)

    num_rows = (max_row - min_row)
    num_cols = t.shape[1]

    idxs = t.indices()[:, index_mask]
    vals = t.values()[index_mask]
    return torch.sparse_coo_tensor(idxs, vals, size=(num_rows, num_cols)).coalesce()


@torch.no_grad
def maxsim(X: torch.Tensor, y: torch.Tensor, maximize: bool, chunk_size: int = 2_000) -> torch.Tensor:
    device = X.device
    n_samples = X.shape[0]
    max_sim_v = torch.empty(n_samples, device=device, dtype=X.dtype)
    max_sim_i = torch.empty(n_samples, device=device, dtype=torch.int64)
    
    splits = int(math.ceil(X.shape[0] / chunk_size))
    for i in tqdm.auto.trange(splits, desc=f'maxsim with chunk_size={chunk_size} on {device}', leave=False, colour="red"):
        if i*chunk_size >= n_samples:
            continue
        start, end = i * chunk_size, min((i + 1) * chunk_size, n_samples)
        sub_x = slice_sparse_tensor_rows(X, start, end)
        if DEBUG_MEM: print(f"[maxsim] step {i} cuda mem free/total = {torch.cuda.mem_get_info()}")
        if DEBUG_MEM: print("sub_x.shape:", sub_x.shape, "//", "y.shape:", y.shape)
        sub_sim = sub_x @ y # TODO – Implement sparse max here to save mem!
        sub_sim = sub_sim.to_dense()
        if maximize:
            sub_max_sim_v, sub_max_sim_i = sub_sim.max(dim=-1)
        else:
            sub_max_sim_v, sub_max_sim_i = sub_sim.min(dim=-1)
        del sub_sim
        del sub_x
        torch.cuda.empty_cache() # needs to happen after maxsim for some reason.
        max_sim_v[start: end] = sub_max_sim_v
        max_sim_i[start: end] = sub_max_sim_i

    return max_sim_v, max_sim_i


@torch.no_grad
def kmeans(
        q: torch.Tensor,
        X: torch.Tensor, 
        k: int,
        max_iters: int = 100, 
        tol: float = 1e-4, 
        maximize: bool = True,
        initialization_strategy: str = "kmeans++",
        seed: int = 42
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    if torch.cuda.is_available():
        q = q.cuda()
        X = X.cuda()
    q = q.float()
    X = X.float()
    torch.manual_seed(seed)
    # Initialize centroids randomly
    print(f"kmeans called with k={k} / q.shape={q.shape} X.shape={X.shape}")

    print(f"initializing with strategy [{initialization_strategy}]")
    centroid_idxs = torch.randperm(X.size(0))[:k]
    if initialization_strategy == "kmeans++":
        first_centroid = X[centroid_idxs[0]]
        centroids = [first_centroid]
        centroids_used_mask = torch.zeros(len(X), device=X.device)
        centroids_used_mask[centroid_idxs[0]] = 1
        d = torch.sparse.mm(X, first_centroid[None].T).to_dense().flatten()
        for j in tqdm.trange(1, k, desc="initializing with kmeans++", colour="green"):
            most_recent_centroid = centroids[-1]
            # Compute distances from each datapoints closest centroid
            d2 = torch.sparse.mm(q, most_recent_centroid[None].T).to_dense().flatten()

            # Take the one that is furthest and make it the next centroid
            if maximize:
                d = d.max(d2)
                d = d + (centroids_used_mask * 10e10)
                best_centroid_idx = d.argmin()
                centroids_used_mask[best_centroid_idx] = 1
            else:
                d = d.min(d2)
                d = d + (centroids_used_mask * -10e10)
                best_centroid_idx = d.argmax()
                centroids_used_mask[best_centroid_idx] = 1
            if DEBUG_MEM: tqdm.tqdm.write(f"--> got centroid {best_centroid_idx} with dist: {d[best_centroid_idx]:.3f}")
            centroids.append(X[best_centroid_idx])
        centroids = torch.stack(centroids)
    else:
        # random initialization
        centroids = torch.stack([X[k] for k in centroid_idxs])

    last_centroid_shift = float("inf")
    
    pbar = tqdm.auto.trange(max_iters)
    print("running kmeans on device:", X.device)
    for j in pbar:
        torch.cuda.empty_cache()
        if DEBUG_MEM: print(f"[kmeans] step {j} cuda mem free/total = {torch.cuda.mem_get_info()}")
        _, assignments = maxsim(
            X=q, y=centroids.T, maximize=maximize
        )
        x_idxs = torch.arange(len(X), device=assignments.device)
        idxs = torch.stack([x_idxs, assignments], dim=1)
        vals = torch.ones(len(X), device=assignments.device)
        sparse_assignment_matrix = (
            torch.sparse_coo_tensor(idxs.T, vals, torch.Size([len(X), k]))
        ).float()
        cluster_idxs = torch.arange(k, device=X.device)
        cluster_idxs, cluster_counts = assignments.unique(return_counts=True)
        num_assignments = torch.zeros((k,), dtype=cluster_idxs.dtype, device=cluster_idxs.device) - 1
        # num_assignments[cluster_idxs] += cluster_counts
        num_assignments.scatter_add_(dim=0, index=cluster_idxs, src=(cluster_counts + 1))
        # num_assignments = (assignments[:, None] == cluster_idxs).sum(0)
        
        centroid_sums = torch.sparse.mm(sparse_assignment_matrix.T, X)
        new_centroids = (
           centroid_sums.to_dense().double() / num_assignments[:, None]
        ).float()

        print(f"iteration {j} num centroids used: {(num_assignments >= 0).sum()}/{k}")
        
        # TODO: Handle dead centroids through re-initialization.
        # don't replace a centroid if it got zero
        new_centroids = torch.where(
            (num_assignments < 0)[:, None], centroids.to_dense(), new_centroids.to_dense()
        ).to_sparse_coo()
        
        total_centroid_shift = (centroids - new_centroids).pow(2).sum(1).sum()
        if torch.isnan(total_centroid_shift):
            raise RuntimeError("got NaN during kmeans :(")
        shift_diff = abs(total_centroid_shift - last_centroid_shift)
        last_centroid_shift = total_centroid_shift
        
        centroids = new_centroids
        pbar.set_description(f"Dist: {total_centroid_shift:.2f} / Diff {shift_diff:.4f}")
        
        if shift_diff < tol:
            print(f"stopping early due to tolerance hit. completed {j} iterations.")
            break
    breakpoint()
    return centroids.cpu(), assignments.cpu()
