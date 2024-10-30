from typing import Dict, List, Tuple

import collections
import gc
import math
import os
import pickle

import datasets
import torch
import tqdm

from bm25_pt.bm25 import TokenizedBM25
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

from cde.lib.cluster_faiss import paired_kmeans_faiss

from .dist import (
    get_rank, 
    get_world_size, 
    print0
)
from .embed import embed_with_cache
from .misc import (
    get_cache_location_from_kwargs,
    tqdm_if_main_worker
)
from .tensor import (
    maxsim,
)


# Whether to maximize distances
# between points during clustering.
SHOULD_MAXIMIZE_CLUSTER_DISTANCE_FOR_MODEL = {
    "bm25": True,
    "gtr_base": False,
}


def pad_to_length(t, max_doc_length: int = 256):
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
    print("[embed_for_clustering] called with model", model, "/ keys", document_key, query_key)
    # return torch.randn(len(dataset), 768), torch.randn(len(dataset), 768)
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
        # model = DenseEncoder("sentence-transformers/gtr-t5-base")

        # https://github.com/UKPLab/sentence-transformers/blob/87f4180d7d197d4a471d627afc788b62a81c0214/sentence_transformers/SentenceTransformer.py#L392
        print("[embed_with_cache] computing query embeddings")
        dataset.set_format("pt")
        print("[embed_with_cache] embedding queries")
        query_embeddings = embed_with_cache(
            "sentence-transformers/gtr-t5-base", 
            dataset_fingerprint + "_queries", 
            dataset,
            query_key,
            save_to_disk=True,
            batch_size=1024,
        )
        print("[embed_with_cache] halving query embeddings")
        query_embeddings = query_embeddings["embeds"].half().cpu()
        assert not query_embeddings.isnan().any(), "got nan query embeddings"
    
        if document_key == query_key:
            print(f"[embed_with_cache] repeating corpus_embeddings num_gpus={num_gpus}")
            corpus_embeddings = query_embeddings

        else:
            print(f"[embed_with_cache] computing corpus_embeddings num_gpus={num_gpus}")
            corpus_embeddings = embed_with_cache(
                "sentence-transformers/gtr-t5-base", 
                dataset._fingerprint + "_documents", 
                dataset,
                document_key,
                save_to_disk=True,
                batch_size=1024,
            )
            corpus_embeddings = corpus_embeddings["embeds"].half().cpu()
            print("[embed_with_cache] got corpus embeddings, remapping")

        assert not corpus_embeddings.isnan().any(), "got nan corpus embeddings"

        
        # qidxs = torch.zeros(len(dataset), dtype=torch.long)
        # qidxs[query_output_idxs] = torch.arange(len(dataset), dtype=torch.long)
        # cidxs = torch.zeros(len(dataset), dtype=torch.long)
        # cidxs[corpus_output_idxs] = torch.arange(len(dataset), dtype=torch.long)
        # ##################################################################
        # query_embeddings = query_embeddings[qidxs]
        # corpus_embeddings = corpus_embeddings[cidxs]
        avg_sim = (corpus_embeddings[:100] @ query_embeddings[:100].T).diag().mean()
        print(f"[embed_with_cache] returning all embeddings => avg_sim={avg_sim:.2f}")
        return query_embeddings, corpus_embeddings
    
    elif model == "bert":
        assert query_to_doc
        dataset_fingerprint = dataset._fingerprint
        num_gpus = torch.cuda.device_count()
        if get_rank() == 0:
            print(f"Embedding {len(dataset)} queries with {num_gpus} GPUs...")

        # https://github.com/UKPLab/sentence-transformers/blob/87f4180d7d197d4a471d627afc788b62a81c0214/sentence_transformers/SentenceTransformer.py#L392
        print("[embed_with_cache] computing query embeddings")
        dataset.set_format("pt")
        print("[embed_with_cache] embedding queries")
        query_embeddings = embed_with_cache(
            "bert-base-uncased", 
            dataset_fingerprint + "_queries", 
            dataset,
            query_key,
            save_to_disk=True,
            batch_size=2048,
        )
        print("[embed_with_cache] halving query embeddings")
        query_embeddings = query_embeddings["embeds"].half().cpu()
        assert not query_embeddings.isnan().any(), "got nan query embeddings"
    
        if document_key == query_key:
            print(f"[embed_with_cache] repeating corpus_embeddings num_gpus={num_gpus}")
            corpus_embeddings = query_embeddings

        else:
            print(f"[embed_with_cache] computing corpus_embeddings num_gpus={num_gpus}")
            corpus_embeddings = embed_with_cache(
                "bert-base-uncased", 
                dataset._fingerprint + "_documents", 
                dataset,
                document_key,
                save_to_disk=True,
                batch_size=2048,
            )
            corpus_embeddings = corpus_embeddings["embeds"].half().cpu()
            print("[embed_with_cache] got corpus embeddings, remapping")

        assert not corpus_embeddings.isnan().any(), "got nan corpus embeddings"
        
        # qidxs = torch.zeros(len(dataset), dtype=torch.long)
        # qidxs[query_output_idxs] = torch.arange(len(dataset), dtype=torch.long)
        # cidxs = torch.zeros(len(dataset), dtype=torch.long)
        # cidxs[corpus_output_idxs] = torch.arange(len(dataset), dtype=torch.long)
        # ##################################################################
        # query_embeddings = query_embeddings[qidxs]
        # corpus_embeddings = corpus_embeddings[cidxs]
        avg_sim = (corpus_embeddings[:100] @ query_embeddings[:100].T).diag().mean()
        print(f"[embed_with_cache] returning all embeddings => avg_sim={avg_sim:.2f}")
        return query_embeddings, corpus_embeddings
    
    elif model == "mixedbread":
        assert query_to_doc
        dataset_fingerprint = dataset._fingerprint
        num_gpus = torch.cuda.device_count()
        if get_rank() == 0:
            print(f"Embedding {len(dataset)} queries with {num_gpus} GPUs...")

        # https://github.com/UKPLab/sentence-transformers/blob/87f4180d7d197d4a471d627afc788b62a81c0214/sentence_transformers/SentenceTransformer.py#L392
        print("[embed_with_cache] computing query embeddings")
        dataset.set_format("pt")
        print("[embed_with_cache] embedding queries")
        query_embeddings = embed_with_cache(
            "mixedbread-ai/mxbai-embed-large-v1",
            dataset_fingerprint + "_queries", 
            dataset,
            query_key,
            save_to_disk=True,
            prefix="Represent this sentence for searching relevant passages: ",
            normalize_embeds=True,
            batch_size=1024,
        )
        print("[embed_with_cache] halving query embeddings")
        query_embeddings = query_embeddings["embeds"].half().cpu()
        assert not query_embeddings.isnan().any(), "got nan query embeddings"
    
        if document_key == query_key:
            print(f"[embed_with_cache] repeating corpus_embeddings num_gpus={num_gpus}")
            corpus_embeddings = query_embeddings

        else:
            print(f"[embed_with_cache] computing corpus_embeddings num_gpus={num_gpus}")
            corpus_embeddings = embed_with_cache(
            "mixedbread-ai/mxbai-embed-large-v1",
                dataset_fingerprint + "_documents", 
                dataset,
                document_key,
                save_to_disk=True,
                prefix="",
                normalize_embeds=True,
                batch_size=1024,
            )
            corpus_embeddings = corpus_embeddings["embeds"].half().cpu()
            print("[embed_with_cache] got corpus embeddings, remapping")

        assert not corpus_embeddings.isnan().any(), "got nan corpus embeddings"
        
        # qidxs = torch.zeros(len(dataset), dtype=torch.long)
        # qidxs[query_output_idxs] = torch.arange(len(dataset), dtype=torch.long)
        # cidxs = torch.zeros(len(dataset), dtype=torch.long)
        # cidxs[corpus_output_idxs] = torch.arange(len(dataset), dtype=torch.long)
        # ##################################################################
        # query_embeddings = query_embeddings[qidxs]
        # corpus_embeddings = corpus_embeddings[cidxs]
        avg_sim = (corpus_embeddings[:100] @ query_embeddings[:100].T).diag().mean()
        print(f"[embed_with_cache] returning all embeddings => avg_sim={avg_sim:.2f}")
        return query_embeddings, corpus_embeddings
    elif model == "sbert":
        assert query_to_doc
        dataset_fingerprint = dataset._fingerprint
        num_gpus = torch.cuda.device_count()
        if get_rank() == 0:
            print(f"Embedding {len(dataset)} queries with {num_gpus} GPUs...")

        # https://github.com/UKPLab/sentence-transformers/blob/87f4180d7d197d4a471d627afc788b62a81c0214/sentence_transformers/SentenceTransformer.py#L392
        print("[embed_with_cache] computing query embeddings")
        dataset.set_format("pt")
        print("[embed_with_cache] embedding queries")
        query_embeddings = embed_with_cache(
            "sentence-transformers/all-MiniLM-L6-v2", 
            dataset_fingerprint + "_queries", 
            dataset,
            query_key,
            save_to_disk=True,
            batch_size=1024,
        )
        print("[embed_with_cache] halving query embeddings")
        query_embeddings = query_embeddings["embeds"].half().cpu()
        assert not query_embeddings.isnan().any(), "got nan query embeddings"
    
        if document_key == query_key:
            print(f"[embed_with_cache] repeating corpus_embeddings num_gpus={num_gpus}")
            corpus_embeddings = query_embeddings

        else:
            print(f"[embed_with_cache] computing corpus_embeddings num_gpus={num_gpus}")
            corpus_embeddings = embed_with_cache(
                "sentence-transformers/all-MiniLM-L6-v2",
                dataset._fingerprint + "_documents", 
                dataset,
                document_key,
                save_to_disk=True,
                batch_size=1024,
            )
            corpus_embeddings = corpus_embeddings["embeds"].half().cpu()
            print("[embed_with_cache] got corpus embeddings, remapping")

        assert not corpus_embeddings.isnan().any(), "got nan corpus embeddings"
        
        # qidxs = torch.zeros(len(dataset), dtype=torch.long)
        # qidxs[query_output_idxs] = torch.arange(len(dataset), dtype=torch.long)
        # cidxs = torch.zeros(len(dataset), dtype=torch.long)
        # cidxs[corpus_output_idxs] = torch.arange(len(dataset), dtype=torch.long)
        # ##################################################################
        # query_embeddings = query_embeddings[qidxs]
        # corpus_embeddings = corpus_embeddings[cidxs]
        avg_sim = (corpus_embeddings[:100] @ query_embeddings[:100].T).diag().mean()
        print(f"[embed_with_cache] returning all embeddings => avg_sim={avg_sim:.2f}")
        return query_embeddings, corpus_embeddings
    else:
        raise ValueError(f"model {model} not supported")


@torch.no_grad
def paired_kmeans_sparse(
        q: torch.Tensor,
        X: torch.Tensor, 
        k: int,
        max_iters: int = 25, 
        tol: float = 1e-3, 
        maximize: bool = True,
        initialization_strategy: str = "kmeans++", # ["kmeans++", "random"]
        seed: int = 42,
        debug_mem_usage: bool = False,
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
            if debug_mem_usage: tqdm.tqdm.write(f"--> got centroid {best_centroid_idx} with dist: {d[best_centroid_idx]:.3f}")
            centroids.append(X[best_centroid_idx])
        centroids = torch.stack(centroids)
    else:
        # random initialization
        centroids = torch.stack([X[k] for k in centroid_idxs])

    last_centroid_shift = float("inf")
    
    pbar = tqdm_if_main_worker(range(max_iters), desc="KMEANS ITER")
    for j in pbar:
        torch.cuda.empty_cache()
        if debug_mem_usage: print(f"[kmeans] step {j} cuda mem free/total = {torch.cuda.mem_get_info()}")
        sims, assignments = maxsim(
            X=q, y=centroids.T, maximize=maximize,
            debug_mem_usage=debug_mem_usage,
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


def cluster_dataset_uncached(
        dataset: datasets.Dataset, 
        query_to_doc: bool,
        model: str,
        query_key: str,
        document_key: str,
        cluster_size: int,
        downscale_and_normalize: bool,
    ) -> Dict[int, List[int]]:
    print0("[cluster_dataset_uncached] calling embed_for_clustering...")
    q, X = embed_for_clustering(
        dataset=dataset,
        query_key=query_key,
        document_key=document_key,
        model=model,
        query_to_doc=query_to_doc,
    )

    if downscale_and_normalize:
        model = Pipeline(
            [
                ('svd', TruncatedSVD(n_components=128, random_state=42)),
                ('normalizer', Normalizer(copy=False))
            ]
        )
        print("[cluster_dataset_uncached] calling fit on queries")
        q_small = q[:100_000, :].cpu().numpy()
        model.fit(q_small)
        print("[cluster_dataset_uncached] calling transform on queries")
        q = model.transform(q.cpu().numpy())
        q = torch.tensor(q, dtype=torch.float16, device='cpu')
        print("[cluster_dataset_uncached] calling transform on docs")
        X = model.transform(X)
        X = torch.tensor(X, dtype=torch.float16, device='cpu')

    gc.collect()
    
    k = math.ceil(len(X) / cluster_size)
    is_sparse = (model == "bm25")

    q = q.cpu()
    X = X.cpu()

    print0("cluster_dataset_uncached] calling faiss...")
    if is_sparse:
        _, assignments = paired_kmeans_sparse(
            q=q, 
            X=X,
            k=k,
            maximize=SHOULD_MAXIMIZE_CLUSTER_DISTANCE_FOR_MODEL[model]
        )
    else:
        _, assignments = paired_kmeans_faiss(
            q=q,
            X=X,
            k=k,
            # maximize=SHOULD_MAXIMIZE_CLUSTER_DISTANCE_FOR_MODEL[model]
        )
    # stack assignments to list.
    assignments_dict = collections.defaultdict(list)
    for i in tqdm.trange(len(assignments), desc="collecting assigments", leave=False):
        assignments_dict[i].append(assignments[i].item())
    return dict(assignments_dict)


def save_cluster_dict(cluster_dict: Dict[int, List[int]], path: str):
    print(f"[save_cluster_dict] Saving cluster dict of length {len(cluster_dict)} / type {type(cluster_dict)} to {path}")
    if isinstance(cluster_dict, list):
        pickle.dump(cluster_dict, open(path.replace(".txt", ".p"), "wb"))
        return
    values = [
            value for key, value in sorted(cluster_dict.items())
    ]
    values = [
        value[0] if isinstance(value, list) else value for value in values
    ]
    with open(path, 'w') as f:
        for item in values:
            f.write("%s\n" % item)
    print(f"Saved dict of {len(cluster_dict)} items to {path}")


def load_cluster_dict(path: str) -> Dict[int, List[int]]:
    print0(f"[load_cluster_dict] loading from {path}")
    if os.path.exists(path.replace(".txt", ".p")):
        return pickle.load(open(path.replace(".txt", ".p"), "rb"))
    
    with open(path, 'r') as f:
        lines = f.readlines()
    result = {}
    for i, line in enumerate(tqdm_if_main_worker(lines, desc="Loading cluster dict from file")):
        result[i] = int(line)
    print(f"Loaded dict of {len(result)} items from {path}")
    return result


def cluster_dataset(
        dataset: datasets.Dataset, 
        query_to_doc: bool,
        model: str,
        query_key: str,
        document_key: str,
        cluster_size: int,
        downscale_and_normalize: bool,
    ) -> Dict[int, List[int]]:
    # TODO: Turn this caching logic into a nice decorator?
    clustering_hash = get_cache_location_from_kwargs(
        # method="cluster_dataset",
        dataset_fingerprint=dataset._fingerprint,
        document_key=document_key,
        query_key=query_key,
        batch_size=cluster_size,
        model=model,
        query_to_doc=query_to_doc,
        downscale_and_normalize=downscale_and_normalize,
    ) + ".txt"
    print0("[cluster_dataset] checking for cluster at file", clustering_hash)
    print0("[cluster_dataset] checking for cluster w args:", 
        f"dataset_fingerprint={dataset._fingerprint}",
        f"document_key={document_key}",
        f"query_key={query_key}",
        f"batch_size={cluster_size}",
        f"model={model}",
        f"query_to_doc={query_to_doc}",
        f"downscale_and_normalize={downscale_and_normalize}"
    )
    if os.path.exists(clustering_hash) or os.path.exists(clustering_hash.replace(".txt", ".p")):
        print("[cluster_dataset] opening cached cluster ... ", clustering_hash)
        result = load_cluster_dict(clustering_hash)
        print("[cluster_dataset] opened cached cluster ... ", clustering_hash)
        return result
    else:
        assert get_world_size() == 1, "can't cluster in DDP"
        MAX_DATASET_LEN = 50_000_000
        if len(dataset) < MAX_DATASET_LEN:
            result = cluster_dataset_uncached(
                dataset=dataset,
                model=model,
                query_to_doc=query_to_doc,
                query_key=query_key,
                document_key=document_key,
                cluster_size=cluster_size,
                downscale_and_normalize=downscale_and_normalize,
            )
        else:
            num_shards = math.ceil(len(dataset) / MAX_DATASET_LEN)
            print(f"[cluster_dataset] splitting into {num_shards} datasets of max length {MAX_DATASET_LEN}")
            dataset = dataset.add_column("sub_idx", range(len(dataset)))
            dataset = dataset.shuffle(seed=42)
            result = {}
            offset = 0
            for i in range(num_shards):
                print(f"[cluster_dataset] selecting shard {i} / {num_shards}")
                mini_dataset = dataset.shard(
                    num_shards=num_shards, 
                    index=i, 
                    contiguous=True, 
                    keep_in_memory=True
                )
                mini_result = cluster_dataset_uncached(
                    dataset=mini_dataset,
                    model=model,
                    query_to_doc=query_to_doc,
                    query_key=query_key,
                    document_key=document_key,
                    cluster_size=cluster_size,
                    downscale_and_normalize=downscale_and_normalize,
                )
                new_clusters = set()
                mini_dataset_sub_idx = mini_dataset["sub_idx"]
                for data_idx, cluster in tqdm_if_main_worker(mini_result.items()):
                    if isinstance(cluster, list): cluster = cluster[0]
                    if isinstance(cluster, torch.Tensor): cluster = cluster.item()
                    new_clusters.add(cluster)
                    true_data_idx = mini_dataset_sub_idx[data_idx]
                    if isinstance(true_data_idx, torch.Tensor): true_data_idx = true_data_idx.item()
                    result[true_data_idx] = cluster + offset
                offset += len(new_clusters)
                gc.collect()
                torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        save_cluster_dict(result, clustering_hash)
        return result


def cluster_subdomains_uncached(
        dataset: datasets.Dataset,
        subdomains: Dict[int, List[int]],
        query_to_doc: bool,
        cluster_size: int, 
        batch_size: int,
        model: str,
        downscale_and_normalize: bool,
    ) -> List[Dict[int, List[int]]]:
    """Creates clusters of cluster_size and combines them into subdomain-specific batches of ~batch_size."""
    if batch_size < cluster_size:
        print0("WARNING: batch size", batch_size, "is less than cluster size", cluster_size)
    
    subdomains_smallest_first = sorted(subdomains.items(), key=lambda x: len(x[1]))
    # subdomains_largest_first = sorted(subdomains.items(), key=lambda x: -len(x[1]))
    all_cluster_assignments = []

    import random
    import time
    random.seed(time.time())
    random.shuffle(subdomains_smallest_first)
    for j, (_, data_idxs) in enumerate(subdomains_smallest_first):
        perc = (j + 1) / len(subdomains) * 100
        print0(f"({j + 1} / {len(subdomains)} -- {perc:.1f}%) selecting {len(data_idxs)} indices for clustering")
        mini_dataset = dataset.dataset.select(data_idxs, keep_in_memory=True)
        print0("[cluster_subdomains] calling cluster_dataset")
        
        # The idea is that we cluster each dataset individually and then track the clusters via
        # nested dicts like this.
        cluster_assignments = cluster_dataset(
            dataset=mini_dataset,
            model=model,
            query_key="query",
            document_key="document",
            query_to_doc=query_to_doc,
            cluster_size=cluster_size,
            downscale_and_normalize=downscale_and_normalize,
        )
        mini_cluster_assignments = collections.defaultdict(list)
        for j, raw_cluster in tqdm_if_main_worker(cluster_assignments.items(), leave=False, colour="blue"):
            if isinstance(raw_cluster, list): raw_cluster = raw_cluster[0]
            if isinstance(raw_cluster, torch.Tensor): raw_cluster = raw_cluster.item()
            mini_cluster_assignments[raw_cluster].append(data_idxs[j])
        all_cluster_assignments.append(dict(mini_cluster_assignments))
    print0(f"[cluster_subdomains] expanded {len(subdomains)} domains to {len(all_cluster_assignments)} clusters.")
    return all_cluster_assignments


def cluster_subdomains(
        dataset: datasets.Dataset,
        subdomains: Dict[int, List[int]],
        query_to_doc: bool,
        batch_size: int, 
        cluster_size: int,
        model: str,
        downscale_and_normalize: bool
    ) -> List[Dict[int, List[int]]]:
    print0("[cluster_subdomains]:", 
        f"dataset_fingerprint={dataset._fingerprint}",
        f"batch_size={batch_size}",
        f"cluster_size={cluster_size}",
        f"model={model}",
        f"query_to_doc={query_to_doc}",
        f"downscale_and_normalize={downscale_and_normalize}",
    )
    # exit()
    clustering_hash = get_cache_location_from_kwargs(
        method="cluster_subdomains__list",
        dataset_fingerprint=dataset._fingerprint,
        batch_size=batch_size,
        cluster_size=cluster_size,
        model=model,
        query_to_doc=query_to_doc,
        downscale_and_normalize=downscale_and_normalize,
    ) + ".txt"
    print0("[cluster_subdomains] checking for cluster at file", clustering_hash)
    if os.path.exists(clustering_hash) or os.path.exists(clustering_hash.replace(".txt", ".p")):
        print0("[cluster_subdomains] opening cached cluster ... ", clustering_hash)
        return load_cluster_dict(clustering_hash)
    else:
        print0("[cluster_subdomains] generating cached cluster ... ", clustering_hash)
        # exit()

        result = cluster_subdomains_uncached(
            dataset=dataset,
            subdomains=subdomains,
            query_to_doc=query_to_doc,
            cluster_size=cluster_size,
            batch_size=batch_size,
            model=model,
            downscale_and_normalize=downscale_and_normalize,
        )
        gc.collect()
        torch.cuda.empty_cache()
        print("[cluster_subdomains] saving result to", clustering_hash)
        save_cluster_dict(result, clustering_hash)
        print("[cluster_subdomains] saved result to", clustering_hash)
        return result
