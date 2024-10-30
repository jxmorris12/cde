from typing import List, Union, Tuple

import sys
sys.path.append('/home/paperspace/tti3')

from typing import List

import pandas as pd
import streamlit as st
import torch
import transformers

from cde.dataset import NomicSupervisedDataset, NomicUnsupervisedDataset
from cde.sampler import (
    RandomSampler, 
    FixedSubdomainSampler, 
    AutoClusterSampler, 
    AutoClusterWithinDomainSampler
)


@st.cache_resource
def load_supervised_dataset() -> NomicSupervisedDataset:        
    tokenizer = transformers.AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1-unsupervised")
    return NomicSupervisedDataset(
        tokenizer=tokenizer, num_hard_negatives=0, max_seq_length=128
    )


@st.cache_resource
def load_unsupervised_dataset() -> NomicUnsupervisedDataset:        
    tokenizer = transformers.AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1-unsupervised")
    return NomicUnsupervisedDataset(
        tokenizer=tokenizer, max_seq_length=128
    )


def get_sampler(
        strategy: str,
        dataset: Union[NomicSupervisedDataset, NomicUnsupervisedDataset],
        query_to_doc: bool,
        batch_size: int,
        model: str,
    )  -> torch.utils.data.Sampler:
    if strategy == "random":
        return RandomSampler(
            dataset=dataset, 
            batch_size=batch_size,
            shuffle=True,
        )
    elif strategy == "domain":
        return FixedSubdomainSampler(
            dataset=dataset, 
            batch_size=batch_size,
            shuffle=False,
        )
    elif strategy == "cluster":
        return AutoClusterSampler(
            dataset=dataset, 
            batch_size=batch_size,
            shuffle=True,
            query_to_doc=query_to_doc, 
            model=model,
        )
    elif strategy == "cluster_within_domain":
        return AutoClusterWithinDomainSampler(
            dataset=dataset, 
            batch_size=batch_size,
            shuffle=True,
            query_to_doc=query_to_doc, 
            model=model,
        )
    

@st.cache_data
def get_sampler_batch_idxs(
        dataset_start_idx: int,
        batch_size: int,
        sampler: torch.utils.data.Sampler
    ) -> List[int]:
    batch_idxs = []
    sampler_iter = iter(sampler)
    for _ in range(dataset_start_idx):
        next(sampler_iter)

    for _ in range(batch_size):
        idx = next(sampler_iter)
        batch_idxs.append(idx)
    return batch_idxs


@st.cache_resource
def get_sampled_batch(
        strategy: str,
        dataset: Union[NomicSupervisedDataset, NomicUnsupervisedDataset],
        batch_size: int,
        model: str,
        dataset_start_idx = 0,
    ) -> Tuple[List, List]:
        sampler = get_sampler(
            strategy=strategy,
            dataset=dataset,
            batch_size=batch_size,
            model=model,
            query_to_doc=True,
        )

        batch_idxs = get_sampler_batch_idxs(
            dataset_start_idx=dataset_start_idx,
            batch_size=batch_size,
            sampler=sampler,
        )

        # get a batch from the sampler
        batch = [dataset.dataset[j] for j in batch_idxs]
        return batch, batch_idxs


def main():
    global dataset_start_idx

    st.title('🎈 TTI Clusters')

    with st.sidebar:
        dataset_name = st.selectbox(
            "Dataset",
            # ["nomic_supervised", "nomic_unsupervised"],
            ["nomic_unsupervised", "nomic_supervised"],
        )
        if dataset_name == "nomic_supervised":
            st.text("[Note: Hard negatives not displayed.]")
        batch_size = st.selectbox(
            "Batch_size",
            [224],
        )
        model = st.selectbox(
            "Model",
            options=["gtr_base", "bm25"],
        )
        sampling_strategy = st.selectbox(
            "Sampler",
            options=[
                "domain", 
                "random", 
                "cluster", 
                "cluster_within_domain"
            ],
        )
    
    with st.spinner('Getting dataset...'):
        if dataset_name == "nomic_unsupervised":
            dataset = load_unsupervised_dataset()
        else:
            dataset = load_supervised_dataset()
    
    # allow toggling through multiple batches
    if 'dataset_start_idx' not in st.session_state:
        st.session_state['dataset_start_idx'] = 0
    if st.button('Reset batch'):
        st.session_state['dataset_start_idx'] = 0
    if st.button('Next batch'):
        st.session_state['dataset_start_idx'] += batch_size
    
    dataset_start_idx = st.session_state['dataset_start_idx']
    st.text(f"Batch index {dataset_start_idx}")

    with st.spinner('Getting samples...'):      
        batch, batch_idxs = get_sampled_batch(
            strategy=sampling_strategy,
            dataset=dataset,
            batch_size=batch_size,
            model=model,
            dataset_start_idx=dataset_start_idx,
        )

    # show it in the UI
    df = pd.DataFrame(batch)
    df["batch_idx"] = batch_idxs

    max_len = 200
    truncate_to_length = lambda s: s[:max_len] + ("..." if len(s) > max_len else "")
    df["query"] = df["query"].apply(truncate_to_length)
    df["document"] = df["document"].apply(truncate_to_length)
    if "negative" in df.columns:
        df["negative"] = df["negative"].apply(lambda batch: list(map(truncate_to_length, batch)))
        df = df.drop("negative", axis=1)

    st.header("Data")
    st.table(df)
        
    st.success('Done!')

if __name__ == '__main__':
    main()