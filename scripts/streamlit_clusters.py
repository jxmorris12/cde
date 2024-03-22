from typing import Dict, List

import streamlit as st
import transformers

from dataset import NomicUnsupervisedDataset
from lib import get_cache_location_from_kwargs


def load_default_dataset() -> NomicUnsupervisedDataset:        
    tokenizer = transformers.AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1-unsupervised")
    return NomicUnsupervisedDataset(tokenizer=tokenizer)


def load_cluster(
        dataset: NomicUnsupervisedDataset,
        query_to_doc: bool,
        batch_size: int,
        model: str
    )  -> Dict[int, List[int]]:
    clustering_hash = get_cache_location_from_kwargs(
        dataset_fingerprint=dataset._fingerprint,
        query_key=dataset._document_input_ids_key,
        document_key=dataset._query_input_ids_key,
        batch_size=batch_size,
        model=model,
        query_to_doc=query_to_doc,
    )
    print("checking for cluster at file", clustering_hash)
    

def main():
    st.title('🎈 App Name')
    batch_size = st.selectbox(
        "Batch_size",
        [224],
    )
    model = st.selectbox(
        "Model",
        key="gtr_base"
        ["bm25", "gtr_base"],
    )

    with st.spinner('Getting cluster...'):
        dataset = load_default_dataset()
        cluster = load_cluster(
            dataset=dataset,
            batch_size=batch_size,
            model=model,
            query_to_doc=True,
        )
        
    st.success('Done!')

if __name__ == '__main__':
    main()