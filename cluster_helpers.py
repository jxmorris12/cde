from typing import Dict, Tuple

import functools
import math

import datasets
import torch
import tqdm

from bm25_pt.bm25 import TokenizedBM25

from dataset import embed_with_cache, NomicSupervisedDataset, RedditDataset
from helpers import gather_sum, get_rank, get_world_size, tqdm_if_main_worker


# Whether to maximize distances
# between points during clustering.
SHOULD_MAXIMIZE_CLUSTER_DISTANCE_FOR_MODEL = {
    "bm25": True,
    "gtr_base": False,
}

DEBUG_MEM = False

def pad_to_length(t, max_doc_length: int = 128):
    if len(t) < max_doc_length:
        nz = max_doc_length - len(t) 
        t = torch.cat((t, torch.zeros((nz,))), dim=0)
    return t

def embed_for_clustering(
        dataset: datasets.Dataset, 
        document_key: str,
        query_key: str,
        model: str,
        query_to_doc: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    # return torch.randn(len(dataset), 512), torch.randn(len(dataset), 512)
    if model == "bm25":
        document_input_ids = dataset[document_key]
        document_input_ids = [pad_to_length(t) for t in tqdm_if_main_worker(document_input_ids)]
        if query_to_doc: 
            query_input_ids = dataset[query_key]
            query_input_ids = [pad_to_length(t) for t in tqdm_if_main_worker(query_input_ids)]
        else:
            query_input_ids = document_input_ids
        document_input_ids = torch.stack(document_input_ids)
        query_input_ids = torch.stack(query_input_ids)
        vocab_size = int(document_input_ids.max() + 1)
        bm25 = TokenizedBM25(vocab_size=vocab_size)
        bm25.index(document_input_ids)
        queries = bm25.docs_to_bags(query_input_ids).to_sparse_coo()
        corpus = bm25._corpus_scores.to_sparse_coo()
        return queries, corpus
    elif model == "gtr_base":
        assert query_to_doc
        dataset_fingerprint = dataset._fingerprint
        num_gpus = torch.cuda.device_count()
        if get_rank() == 0:
            print(f"Embedding {len(dataset)} queries with {num_gpus} GPUs...")

        from dataset import DenseEncoder
        model = DenseEncoder( "sentence-transformers/gtr-t5-base")

        # https://github.com/UKPLab/sentence-transformers/blob/87f4180d7d197d4a471d627afc788b62a81c0214/sentence_transformers/SentenceTransformer.py#L392
        print("computing query embeddings")

        def add_length_columns(ex: Dict) -> Dict:
            ex["query_length"] = list(map(len, ex["query"]))
            ex["document_length"] = list(map(len, ex["document"]))
            return ex

        dataset.set_format("pt")

        dataset = dataset.add_column("idx", range(len(dataset)))
        print("[embed] computing lengths")
        dataset = dataset.map(add_length_columns, batched=True)
        print("[embed] flattening")
        dataset = dataset.flatten_indices()
        print("[embed] sorting by query length")
        dataset = dataset.sort("query_length")
        
        print("[embed] embedding queries")
        query_embeddings = embed_with_cache(
            "sentence-transformers/gtr-t5-base", 
            dataset_fingerprint + "_queries", 
            dataset,
            "query",
            save_to_disk=False,
            model=model,
        )
        print("halving query embeddings")
        query_embeddings = query_embeddings["embeds"].half()
        assert not query_embeddings.isnan().any(), "got nan query embeddings"
        

        print("[embed] sorting by doc length")
        query_output_idxs = dataset["idx"]
        dataset = dataset.sort("document_length")
        corpus_output_idxs = dataset["idx"]
    
        print(f"[embed_with_cache] computing corpus_embeddings num_gpus={num_gpus}")
        corpus_embeddings = embed_with_cache(
            "sentence-transformers/gtr-t5-base", 
            dataset._fingerprint + "_documents", 
            dataset,
            "document",
            # save_to_disk=True,
            save_to_disk=False,
            model=model,
        )
        corpus_embeddings = torch.tensor(
            corpus_embeddings["embeds"]).half()
        print("[embed_with_cache] got corpus embeddings")

        assert not corpus_embeddings.isnan().any(), "got nan corpus embeddings"

        print("[embed_with_cache] remapping embeddings")
        qidxs = []
        cidxs = []
        for i in tqdm.trange(len(dataset), desc='remapping embeddings', colour='MAGENTA', leave=False):
            # TODO: vectorize this
            #       (https://discuss.pytorch.org/t/reverse-inverse-indices-torch-unique/114521/5?u=jxmorris12)
            qidxs.append((query_output_idxs == i).int().argmax())
            cidxs.append((corpus_output_idxs == i).int().argmax())
        query_embeddings = query_embeddings[qidxs]
        corpus_embeddings = corpus_embeddings[cidxs]
        print("returning all embeddings")
        return query_embeddings, corpus_embeddings
    else:
        raise ValueError(f"model {model} not supported")


def slice_sparse_tensor_rows(t: torch.sparse.Tensor, min_row: int, max_row: int) -> torch.sparse.Tensor:
    assert min_row < max_row, f"can't slice from row {min_row} to {max_row}"
    t = t.coalesce()
    row_idxs = t.indices()[0]
    index_mask = (min_row <= row_idxs) & (row_idxs < max_row)

    num_rows = (max_row - min_row)
    num_cols = t.shape[1]

    idxs = t.indices()[:, index_mask]
    vals = t.values()[index_mask]
    return torch.sparse_coo_tensor(idxs, vals, size=(num_rows, num_cols)).coalesce()


def slice_tensor_rows(t: torch.Tensor, min_row: int, max_row: int) -> torch.Tensor:
    if t.is_sparse:
        return slice_sparse_tensor_rows(t=t, min_row=min_row, max_row=max_row)
    else:
        return t[min_row:max_row]


@torch.no_grad
def maxsim(X: torch.Tensor, y: torch.Tensor, maximize: bool, chunk_size: int = 8_000) -> torch.Tensor:
    device = X.device
    n_samples = X.shape[0]

    max_sim_v = torch.zeros(n_samples, device=device, dtype=X.dtype)
    max_sim_i = torch.zeros(n_samples, device=device, dtype=torch.int64)

    # TODO: Implement faster max (without going to dense tensors).
    # TODO: Use multiple GPUs.
    rank = get_rank()
    world_size = get_world_size()

    worker_worklist_size = int(math.ceil(n_samples / world_size))
    splits_start_idx = worker_worklist_size * rank
    splits_end_idx = worker_worklist_size * (rank + 1)

    for i in range(splits_start_idx, splits_end_idx, chunk_size):
        start, end = i, min(i + chunk_size, n_samples)
        sub_x = slice_tensor_rows(X, start, end)
        if DEBUG_MEM: print(f"[maxsim] step {i} cuda mem free/total = {torch.cuda.mem_get_info()}")
        if DEBUG_MEM: print("sub_x.shape:", sub_x.shape, "//", "y.shape:", y.shape)
        sub_sim = sub_x @ y # TODO – Implement sparse max here to save mem!
        sub_sim = sub_sim
        if maximize:
            sub_max_sim_v, sub_max_sim_i = sub_sim.to_dense().max(dim=-1)
        else:
            sub_max_sim_v, sub_max_sim_i = sub_sim.to_dense().min(dim=-1)
        del sub_sim
        del sub_x
        torch.cuda.empty_cache() # needs to happen after maxsim for some reason.
        max_sim_v[start: end] = sub_max_sim_v
        max_sim_i[start: end] = sub_max_sim_i
    
    # gather
    max_sim_v = gather_sum(max_sim_v)
    max_sim_i = gather_sum(max_sim_i)
    k = y.shape[1]

    assert max_sim_v.shape == (n_samples,)
    assert max_sim_i.shape == (n_samples,)
    assert max_sim_i.min() >= 0
    assert max_sim_i.max() <= k

    return max_sim_v, max_sim_i


@torch.no_grad
def paired_kmeans(
        q: torch.Tensor,
        X: torch.Tensor, 
        k: int,
        max_iters: int = 100, 
        tol: float = 1e-3, 
        maximize: bool = True,
        initialization_strategy: str = "kmeans++", # ["kmeans++", "random"]
        seed: int = 42
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Runs paired kmeans.

    Returns:
        - centroids – float torch.Tensor of shape (k, d) containing coordinates of discovered centroids
        – assignments – long torch.Tensor of shape (len(q),) containing cluster indices of each datapoint in q

    """
    if torch.cuda.is_available():
        q = q.cuda()
        X = X.cuda()
    q = q.float()
    X = X.float()
    torch.manual_seed(seed)

    # Initialize centroids randomly
    if get_rank() == 0:
        print(f"]kmeans] called with k={k} / q.shape={q.shape} X.shape={X.shape} / world_size = {get_world_size()}")
        print(f"[kmeans] initializing with strategy [{initialization_strategy}]")
    g = torch.Generator()
    g.manual_seed(seed)
    centroid_idxs = torch.randperm(X.size(0), generator=g)[:k]
    if initialization_strategy == "kmeans++":
        first_centroid = X[centroid_idxs[0]]
        centroids = [first_centroid]
        centroids_used_mask = torch.zeros(len(X), device=X.device)
        centroids_used_mask[centroid_idxs[0]] = 1
        d = torch.sparse.mm(X, first_centroid[None].T).to_dense().flatten()
        for j in tqdm_if_main_worker(range(1, k), desc="initializing with kmeans++", colour="green"):
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
    
    pbar = tqdm_if_main_worker(range(max_iters), desc="KMEANS ITER")
    for j in pbar:
        torch.cuda.empty_cache()
        if DEBUG_MEM: print(f"[kmeans] step {j} cuda mem free/total = {torch.cuda.mem_get_info()}")
        sims, assignments = maxsim(
            X=q, y=centroids.T, maximize=maximize
        )
        avg_sim = sims.mean()
        x_idxs = torch.arange(len(X), device=assignments.device)
        idxs = torch.stack([x_idxs, assignments], dim=1)
        vals = torch.ones(len(X), device=assignments.device)
        sparse_assignment_matrix = (
            torch.sparse_coo_tensor(idxs.T, vals, torch.Size([len(X), k]))
        ).float()
        cluster_idxs = torch.arange(k, device=X.device)
        cluster_idxs, cluster_counts = assignments.unique(return_counts=True)
        num_assignments = torch.zeros((k,), dtype=cluster_idxs.dtype, device=cluster_idxs.device) - 1
        num_assignments.scatter_add_(dim=0, index=cluster_idxs, src=(cluster_counts + 1))
        
        centroid_sums = torch.sparse.mm(sparse_assignment_matrix.T, X)
        new_centroids = (
           centroid_sums.to_dense().double() / num_assignments[:, None]
        ).float()

        print(f"iteration {j} num centroids used: {(num_assignments >= 0).sum()}/{k}")
        
        # TODO: compute & enforce elbow criterion.
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
        if get_rank() == 0:
            pbar.set_description(f"Sim: {avg_sim} / Shift: {total_centroid_shift:.2f} / Diff {shift_diff:.4f}")
        
        if shift_diff < tol:
            print(f"stopping early due to tolerance hit. completed {j} iterations.")
            break
    
    if get_rank() == 0:
        print("exiting kmeans with total shift", total_centroid_shift.item())
    return centroids.cpu(), assignments.cpu()

