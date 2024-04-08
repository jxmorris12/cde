import sys
sys.path.append('/home/paperspace/tti3')

import re

import pytest
import transformers

from dataset import (
    NomicSupervisedDataset, 
    NomicUnsupervisedDataset
)


@pytest.fixture
def tokenizer():
    return transformers.AutoTokenizer.from_pretrained("bert-base-uncased")


def test_one_row_nomic_unsupervised(tokenizer):
    ds = NomicUnsupervisedDataset(
        tokenizer=tokenizer,
        max_seq_length=32,
        use_prefix=True,
    )
    ex = ds[0]
    assert ex["query"].startswith("clustering:")
    assert ex["document"].startswith("clustering:")
    ds.use_prefix = False
    ex_noprefix = ds[0]
    assert not ex_noprefix["query"].startswith("clustering:")
    assert not ex_noprefix["document"].startswith("clustering:")

def test_one_row_nomic_unsupervised(tokenizer):
    ds = NomicSupervisedDataset(
        tokenizer=tokenizer,
        max_seq_length=16,
        use_prefix=True,
    )
    pattern = r'^[a-zA-Z_]+: .+$'
    for i in range(64):
        ex = ds[i]
        assert re.match(pattern, ex["query"])
        assert re.match(pattern, ex["document"])


def test_one_row_nomic_supervised(tokenizer):
    ds = NomicSupervisedDataset(
        tokenizer=tokenizer,
        max_seq_length=32,
        use_prefix=True,
    )
    ex = ds[0]
    assert ex["query"].startswith("search_query: ")
    assert ex["document"].startswith("search_document: ")

    ds.use_prefix = False
    ex_noprefix = ds[0]
    assert not ex_noprefix["query"].startswith("search_query:")
    assert not ex_noprefix["document"].startswith("search_document:")


def test_multirow_nomic_supervised(tokenizer):
    ds = NomicSupervisedDataset(
        tokenizer=tokenizer,
        max_seq_length=64,
        use_prefix=True,
    )
    pattern = r'^[a-zA-Z_]+: .+$'
    for i in range(128):
        ex = ds[i]
        assert re.match(pattern, ex["query"])
        assert re.match(pattern, ex["document"])

