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


def check_available_ram(device="cpu") -> int:
  """
  Returns available RAM on target device
  args:
    device:     str or torch.device
  """
  if isinstance(device, str):
    device = torch.device(device)
  elif isinstance(device, torch.device):
    device = device
  else:
    raise RuntimeError("`device` must be str or torch.device")
  
  if device.type == "cpu":
    return psutil.virtual_memory().available
  else:
    total = torch.cuda.get_device_properties(device).total_memory
    used = torch.cuda.memory_allocated(device)
    return total - used

def will_it_fit(size, device="cpu", safe_mode=True) -> bool:
  """
  Returns True if an array of given byte size fits in target device.
  if self.safe_mode = False, this function simply compares the given byte size with the remaining RAM on target device. This option is faster, 
    but it doesn't take memory fragmentation into account. So it will still be possible to run out of memory.
  if self.safe_mode = True, it will try to allocate a tensor with the given size. if allocation fails, return False. 
    This option is recommended when the other option fails because of OOM.
  
  args:
    size:       int
    device:     str or torch.device
    safe_mode:  bool
  returns:
    result:     bool
  """
  if safe_mode:
    try:
      torch.empty(size, device=device, dtype=torch.uint8)
    except:
      return False
    return True
  else:
    return check_available_ram(device) >= size


def find_optimal_splits(n: int, get_required_memory: Callable, device="cpu") -> int:
  """
  Find an optimal number of split for `n`, such that `get_required_memory(math.ceil(n / n_split))` fits in target device's RAM.
  get_required_memory should be a function that receives `math.ceil(n/n_split)` and returns the required memory in bytes.
  args:
      n:                      int
      get_required_memory:    function
      device:                 str or torch.device
  returns:
      n_splits:               int
  """
    #   https://github.com/DeMoriarty/fast_pytorch_kmeans/blob/master/fast_pytorch_kmeans/util.py
  splits = 1
  sub_n = n
  break_next = False
  while True:
    if break_next:
      break
    if splits > n:
      splits = n
      break_next = True
    sub_n = math.ceil(n / splits)
    required_memory = get_required_memory(sub_n)
    if will_it_fit(required_memory, device):
      break
    else:
      splits *= 2
      continue
  return splits


def slice_sparse_tensor_rows(t: torch.sparse.Tensor, min_row: int, max_row: int) -> torch.sparse.Tensor:
    row_idxs = t.indices()[0]
    index_mask = (min_row <= row_idxs) & (row_idxs < max_row)

    num_rows = (max_row - min_row)
    num_cols = t.shape[1]

    idxs = t.indices()[:, index_mask]
    vals = t.values()[index_mask]
    return torch.sparse_coo_tensor(idxs, vals, size=(num_rows, num_cols)).coalesce()


@torch.no_grad
def maxsim(X: torch.Tensor, centroids: torch.Tensor, maximize: bool, chunk_size: int = 100_000) -> torch.Tensor:
    device = X.device
    n_samples = X.shape[0]
    max_sim_v = torch.empty(n_samples, device=device, dtype=X.dtype)
    max_sim_i = torch.empty(n_samples, device=device, dtype=torch.int64)
    
    chunk_size = math.ceil(n_samples / splits)

    for i in tqdm.auto.trange(splits, desc=f'maxsim with chunk_size={chunk_size} on {device}', leave=False):
        if i*chunk_size >= n_samples:
            continue
        start, end = i * chunk_size, min((i + 1) * chunk_size, n_samples)
        sub_x = slice_sparse_tensor_rows(X, start, end)
        sub_sim = sub_x @ centroids
        sub_sim = sub_sim.to_dense()
        if maximize:
            sub_max_sim_v, sub_max_sim_i = sub_sim.max(dim=-1)
        else:
            sub_max_sim_v, sub_max_sim_i = sub_sim.min(dim=-1)
        del sub_sim
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
        seed: int = 42
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    if torch.cuda.is_available():
        q = q.cuda()
        X = X.cuda()
    q = q.half()
    X = X.half()
    torch.manual_seed(seed)
    # Initialize centroids randomly
    print(f"kmeans called with k={k} / q.shape={q.shape} X.shape={X.shape}")
    centroid_idxs = torch.randperm(X.size(0))[:k]
    centroids = torch.stack([X[k] for k in centroid_idxs]).half()
    last_centroid_shift = float("inf")
    
    pbar = tqdm.auto.trange(max_iters)
    print("running kmeans on device:", X.device)
    for j in pbar:
        _, assignments = maxsim(
            q, centroids.T, maximize=maximize
        )
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
        ).half()
        
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
