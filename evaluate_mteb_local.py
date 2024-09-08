import faiss # Need to import first for some reason.

import argparse
import os
import random
import time

import datasets

from cde.lib.embed import DenseEncoder
from cde.lib.model_configs import MODEL_FOLDER_DICT
from cde.lib.utils import analyze_utils

from cde.lib.eval.mteb import MTEB



TASK_LIST_RETRIEVAL = [
    "FiQA2018", 
    "TRECCOVID",
    "HotpotQA",
    "ArguAna",
    "ClimateFEVER",
    "CQADupstackAndroidRetrieval",
    "CQADupstackEnglishRetrieval",
    "CQADupstackGamingRetrieval",
    "CQADupstackGisRetrieval",
    "CQADupstackMathematicaRetrieval",
    "CQADupstackPhysicsRetrieval",
    "CQADupstackProgrammersRetrieval",
    "CQADupstackStatsRetrieval",
    "CQADupstackTexRetrieval",
    "CQADupstackUnixRetrieval",
    "CQADupstackWebmastersRetrieval",
    "CQADupstackWordpressRetrieval",
    "DBPedia",
    "FiQA2018",
    "NFCorpus",
    "NQ",
    "QuoraRetrieval",
    "SCIDOCS",
    "SciFact",
    "Touche2020",
    # "TRECCOVID",
    "FEVER",
    "MSMARCO",
]

TASK_LIST_RETRIEVAL = ["SCIDOCS", "SciFact", "NFCorpus", "TRECCOVID"] # Tiny datasets.
# TASK_LIST_RETRIEVAL = ["SCIDOCS", "SciFact", "NFCorpus", "TRECCOVID", "Touche2020"] # Small datasets.
# TASK_LIST_RETRIEVAL = ["TRECCOVID"]
# TASK_LIST_RETRIEVAL = ["NFCorpus"]
# TASK_LIST_RETRIEVAL = ["FiQA2018"]


# TASK_LIST_RETRIEVAL = [
#     "ArguAna",
#     "NFCorpus", 
#     "SCIDOCS", 
#     "TRECCOVID", 
#     "SciFact", 
#     "FiQA2018", 
#     "Touche2020",
# ] # Small datasets.


# TASK_LIST_RETRIEVAL = ["QuoraRetrieval"]
# TASK_LIST_RETRIEVAL = ["NFCorpus"]

def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Process model key")
    parser.add_argument(
        "--model_key", 
        help="The key for the model", 
        type=str, 
        default="cde--filter--5epoch",
        choices=MODEL_FOLDER_DICT.keys(),
    )
    parser.add_argument(
        "--cluster_size", 
        help="Cluster size for evaluation", 
        type=int, 
        default=256,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_folder = MODEL_FOLDER_DICT[args.model_key]
    trainer, (_model_args, data_args, _training_args) = analyze_utils.load_trainer_from_checkpoint_and_args(
        model_folder=model_folder,
        load_from_checkpoint=True,
        return_args=True
    )
    trainer.model.eval()

    datasets.enable_caching()
    mteb_encoder = DenseEncoder(
        model_name_or_path=trainer.model.config.embedder,
        encoder=trainer.model.second_stage_model,
        max_seq_length=trainer.model.config.max_seq_length,
        query_prefix="search_query: " if data_args.use_prefix else "",
        document_prefix="search_document: " if data_args.use_prefix else "",
    )
    first_stage_mteb_encoder = DenseEncoder(
        model_name_or_path=trainer.model.config.embedder,
        encoder=trainer.model.first_stage_model,
        max_seq_length=trainer.model.config.max_seq_length,
        query_prefix="search_query: " if data_args.use_prefix else "",
        document_prefix="search_document: " if data_args.use_prefix else "",
    )

    random.Random(time.time()).shuffle(TASK_LIST_RETRIEVAL)
    for task_idx, task in enumerate(TASK_LIST_RETRIEVAL):
        print(f"Beginning {task} ({task_idx+1} / {len(TASK_LIST_RETRIEVAL)})")
        evaluation = MTEB(
            tasks=[task], 
            task_langs=["en"],
            cluster_embedder="sentence-transformers/gtr-t5-base",
            cluster_size=256,
        )
        split = "dev" if task == "MSMARCO" else "test"
        random.seed(42)
        ##################################################
        results = evaluation.run(
            mteb_encoder, 
            first_stage_model=first_stage_mteb_encoder,
            output_folder=os.path.join("results_mteb", args.model_key, str(args.cluster_size)),
            batch_size=512, 
            # corpus_chunk_size=500,
            # corpus_chunk_size=10_000,
            corpus_chunk_size=50_000,
            verbosity=2,
            eval_splits=[split]
        )
        print(task)
        print("\t", results)
        if len(results):
            print("NDCG@10 =>", results[task][split]['ndcg_at_10'])
        print()
    

if __name__ == '__main__':
    main()
