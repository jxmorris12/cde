from dataset import NomicDataset
d = NomicDataset()

import torch
corpus_input_ids = (d.dataset["document_input_ids_t5"])

import tqdm

def ptl(t):
    if len(t) < 64:
        nz = 64 - len(t) 
        t = torch.cat((t, torch.zeros((nz,))), dim=0)
    return t
corpus_input_ids = [ptl(t) for t in tqdm.auto.tqdm(corpus_input_ids)]
corpus_input_ids = torch.stack(corpus_input_ids)

from bm25_pt.bm25 import TokenizedBM25

vocab_size = int(corpus_input_ids.max().item())
breakpoint()
bm25 = TokenizedBM25(vocab_size=vocab_size)
bm25.index(corpus_input_ids)

import transformers
tok = transformers.AutoTokenizer.from_pretrained('t5-base')

idxs = bm25._corpus.indices()
vals = bm25._corpus.values().clamp(max=1)
doc_freqs = torch.sparse_coo_tensor(
    idxs, 
    vals, 
    size=bm25._corpus.shape
).coalesce()

n_subsample = 10_000
random_indices = torch.randperm(len(idxs))[:n_subsample]
minicorpus = torch.sparse_coo_tensor(
    idxs[:, random_indices], 
    vals[random_indices], 
    size=(n_subsample, bm25._corpus.shape[1])
).coalesce()
minicorpus_doc_freqs = torch.sparse_coo_tensor(
    idxs[:, :n_subsample], 
    vals[:n_subsample].clamp(max=1), 
    size=(n_subsample, bm25._corpus.shape[1])
).coalesce()

# try combinatorial solver here
import cvxpy as cp
import math
import scipy

corpus_size = minicorpus.shape[0]

def sparse_batch_mean(st: torch.sparse.Tensor) -> torch.sparse.Tensor:
    num_rows = st.shape[0]
    rows = torch.ones((1, num_rows,), dtype=torch.float32)
    print(rows.shape, st.shape)
    return (rows.to_sparse().coalesce() @ st.float()) / num_rows

def sparse_log(st: torch.sparse.Tensor) -> torch.sparse.Tensor:
    vals = st.values().log()
    idxs = st.indices()
    return torch.sparse_coo_tensor(idxs, vals, size=st.shape).coalesce()

def torch_sparse_to_scipy(t: torch.sparse.Tensor):
    indices = t.coalesce().indices()
    values = t.coalesce().values()
    size = t.size()
    return scipy.sparse.coo_matrix((values.numpy(), (indices[0].numpy(), indices[1].numpy())), shape=size)

minicorpus_avg_idf = -sparse_log(sparse_batch_mean(minicorpus))
minicorpus_avg_idf_exp = -(sparse_batch_mean(minicorpus)).to_dense().squeeze().exp().numpy()

print("doc_freqs.shape:", doc_freqs.shape)
print("minicorpus_avg_idf.shape:", minicorpus_avg_idf.shape)

x = cp.Variable(corpus_size, boolean=True)
# cost = cp.sum_squares(
#     cp.log(x @ torch_sparse_to_scipy(minicorpus_doc_freqs)) # - (minicorpus_avg_idf.to_dense().squeeze().numpy())
# )
cost = cp.sum_squares(
    (x @ torch_sparse_to_scipy(minicorpus_doc_freqs)) - (minicorpus_avg_idf_exp)
)
objective = cp.Minimize(cost)
constraints = [cp.sum(x) == 128]
prob = cp.Problem(objective, constraints)
prob.solve() # solver=cvxpy.CVXOPT)

breakpoint()
