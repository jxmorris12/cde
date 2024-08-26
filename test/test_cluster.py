from typing import Dict

import pytest

import datasets
import numpy as np
import torch
import transformers

from cde.lib import cluster_dataset, paired_kmeans_faiss


@pytest.fixture
def tiny_dataset() -> datasets.Dataset:
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = datasets.load_dataset("rotten_tomatoes")
    dataset = dataset["train"]
    dataset = dataset.select(range(0, 64))
    def tokenize(ex: Dict) -> Dict:
        return tokenizer(
            ex["text"], 
            truncation=True, 
            padding=True,
            max_length=32
        )
    dataset = dataset.map(
        tokenize, batched=True, keep_in_memory=True)
    dataset.set_format("pt")

    dataset.set_format(type=None, columns=["text"])
    dataset = dataset.add_column("query", dataset["text"])
    dataset = dataset.add_column("document", dataset["text"])

    return dataset


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

@pytest.mark.parametrize(
        "query_to_doc, model", [
            (True, "bm25"),
            (False, "bm25"),
            (True, "gtr_base")
        ]
)
def test_cluster_bm25(
        tiny_dataset: datasets.Dataset,
        query_to_doc: bool, 
        model: str
    ):
    batch_size = 2
    clusters = cluster_dataset(
        dataset=tiny_dataset,
        query_to_doc=query_to_doc,
        model=model,
        query_key="input_ids",
        document_key="input_ids",
        batch_size=batch_size
    )
    assert clusters is not None
    all_clusters = np.array(list(clusters.values())).flatten()
    unique, counts = np.unique(all_clusters, return_counts=True)

    # assert there are as many clusters as we specified
    assert len(unique) == len(tiny_dataset) / batch_size
    # check that they are a little imbalanced
    assert 1.0 < counts.mean() <= batch_size
