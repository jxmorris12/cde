import sys
sys.path.append('/home/paperspace/tti3')

import pytest
import transformers

from dataset import (
    BeirDataset,
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


def test_one_row_nomic_supervised():
    ds = NomicSupervisedDataset(
        tokenizer=tokenizer,
        max_seq_length=32,
        use_prefix=True,
    )
    ex = ds[0]
    breakpoint()
    assert ex["query"].startswith("clustering:")
    assert ex["document"].startswith("clustering:")
    ds.use_prefix = False
    ex_noprefix = ds[0]
    assert not ex_noprefix["query"].startswith("clustering:")
    assert not ex_noprefix["document"].startswith("clustering:")


def test_one_row_beir():
    ds = BeirDataset(
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