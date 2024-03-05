from typing import Tuple

import torch
import tqdm

from bm25_pt.bm25 import TokenizedBM25


# Whether to maximize distances
# between points during clustering.
SHOULD_MAXIMIZE_CLUSTER_DISTANCE_FOR_MODEL = {
    "bm25": True,
    "nomic_embed": False,
}


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
        corpus = bm25._corpus.to_sparse_coo()
        return queries, corpus
    elif model == "nomic_embed":
        # TODO compute embeddings here
        raise NotImplementedError()
    else:
        raise ValueError(f"model {model} not supported")


def kmeans(
        q: torch.Tensor,
        X: torch.Tensor, 
        k: int,
        max_iters: int = 100, 
        tol: float = 1e-4, 
        equal: bool = False,
        maximize: bool = True,
        initialization: str = "kmeans++", # ["kmeans++", "random"]
        seed: int = 42
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    # Initialize centroids randomly

    print(f"initializing with algorithm [{initialization}]")
    if initialization == "random":
        centroid_idxs = torch.randperm(X.size(0))[:k]
        centroids = torch.stack([X[k] for k in centroid_idxs]).to_dense()
    else:
        # placeholder
        centroid_idxs = torch.randperm(X.size(0))[:k]

    last_centroid_shift = float("inf")
    
    pbar = tqdm.auto.trange(max_iters)
    print("running kmeans on device:", X.device)
    for j in pbar:
        # Assign each point to the nearest centroid
        distances = (q @ centroids.T).to_dense()
        if equal:
            # equal cluster-sized KMeans can just involve solving the bipartite 
            # matching problem at each step
            # # TODO: use linear_sum_assignment or another matching algorithm?
            # assignments = torch.zeros(len(X))
            # i = 0
            # while i < len(X):
            #     assignments[
            raise NotImplementedError()
        else:
            if maximize:
                _, assignments = distances.max(dim=1)
            else:
                _, assignments = distances.min(dim=1)

        idxs = torch.tensor([[idx, a] for idx, a in zip(range(len(X)), assignments)])
        vals = torch.ones(len(X))
        sparse_assignment_matrix = (
            torch.sparse_coo_tensor(idxs.T, vals, torch.Size([len(X), k]))
            # torch.sparse.FloatTensor(idxs.T, vals, torch.Size([len(X), k]))
        ).to(X.device)
        cluster_idxs = torch.arange(k, device=X.device)
        num_assignments = (assignments[:, None] == cluster_idxs).sum(0)
        
        centroid_sums = sparse_assignment_matrix.T @ X
        new_centroids = (
           centroid_sums.to_dense().double() / num_assignments[:, None]
        ).float()
        
        # don't replace a centroid if it got zero
        new_centroids = torch.where(
            (num_assignments == 0)[:, None], centroids, new_centroids
        )
        
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
    return centroids.cpu(), assignments.cpu()
