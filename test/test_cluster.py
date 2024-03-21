import sys
sys.path.append('/home/paperspace/tti3')

import torch

from helpers import paired_kmeans_faiss


def test_cluster_tiny():
    torch.manual_seed(42)
    d = 8
    A = torch.randn((d,), dtype=torch.float32) * 100
    B = torch.randn((d,), dtype=torch.float32) * 100
    C = torch.randn((d,), dtype=torch.float32)

    c1_size = 6
    c1 = A + torch.randn((c1_size, d), dtype=torch.float32)
    c2_size = 9
    c2 = B + torch.randn((c2_size, d), dtype=torch.float32)
    c3_size = 5
    c3 = C + torch.randn((c3_size, d), dtype=torch.float32)

    points = torch.cat([c1, c2, c3], dim=0)
    _centroids, assignments = paired_kmeans_faiss(
        q=points,
        X=points,
        k=3,
        max_iters=10,
        seed=42
    )
    assignments = torch.tensor(assignments.flatten())
    assert (
        assignments == torch.tensor(
            [2, 2, 2, 2, 2, 2, 
            0, 0, 0, 0, 0, 0, 0, 0, 0, 
            1, 1, 1, 1, 1]
        )
    ).all()

