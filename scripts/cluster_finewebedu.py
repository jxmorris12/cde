import argparse

import transformers

from cde.dataset import FineWeb, FineWebEdu
from cde.lib import cluster_dataset

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster_size", type=int, default=1024)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--model_name", type=str, default="nomic-ai/nomic-bert-2048")
    return parser.parse_args()

def main():
    args = parse_args()

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name)
    dataset = FineWebEdu(
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        tiny=True,
    )
    print(f"[***] Clustering {len(dataset)} docs...")
    cluster_results = cluster_dataset(
        dataset=dataset.dataset,
        model="gtr_base",
        document_key=dataset._document_input_ids_key,
        query_key=dataset._query_input_ids_key,
        query_to_doc=True,
        cluster_size=args.cluster_size,
        downscale_and_normalize=True,
    )
    print("[***] Finished clustering")

if __name__ == "__main__": main()