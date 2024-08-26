import pytest

from cde.dataset import NomicSupervisedDataset
from cde.sampler import FixedSubdomainSampler, RandomSampler

@pytest.fixture
def nomic_dataset():
    dataset = NomicSupervisedDataset(
        num_hard_negatives=1
    )
    return dataset


def test_sampler_random(nomic_dataset):
    sampler = RandomSampler(
        dataset=nomic_dataset,
        batch_size=4,
        shuffle=False,
    )
    idxs = []
    for _ in iter(sampler):
        idxs.append(_)
    assert len(idxs) == len(nomic_dataset)
    assert len(set(idxs)) == len(nomic_dataset)
    assert set(idxs) == set(range(len(nomic_dataset)))


def test_sampler_domain(nomic_dataset):
    batch_size = 4
    sampler = FixedSubdomainSampler(
        dataset=nomic_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    idxs = []
    for _ in iter(sampler):
        idxs.append(_)

    effective_length = len(nomic_dataset) - (len(nomic_dataset) % batch_size)
    assert len(idxs) == effective_length
    assert len(set(idxs)) == effective_length
