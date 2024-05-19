import argparse
import functools
import os
import random

import torch

from spider.lib.embed import DenseEncoder
from spider.lib.model_configs import MODEL_FOLDER_DICT, ARGS_STR_DICT
from spider.lib.utils import analyze_utils

from mteb import MTEB
# from spider.lib.eval.mteb import MTEB
# from spider.lib.eval.mteb.evaluation.evaluators import cos_sim

TASK_LIST_RETRIEVAL = [
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
    "MSMARCO",
    "NFCorpus",
    "NQ",
    "QuoraRetrieval",
    "SCIDOCS",
    "SciFact",
    "Touche2020",
    "TRECCOVID",
    "HotpotQA",
    "FEVER",
]

TASK_LIST_RETRIEVAL = ["SCIDOCS", "SciFact", "NFCorpus", "TRECCOVID", "Touche2020"] # Small datasets.
# TASK_LIST_RETRIEVAL = ["QuoraRetrieval"]
# TASK_LIST_RETRIEVAL = ["TRECCOVID"]
# TASK_LIST_RETRIEVAL = ["FiQA2018"]


TASK_LIST_RETRIEVAL = [
    "FiQA2018", 
    "SCIDOCS", 
    "SciFact", 
    "NFCorpus", 
    "TRECCOVID", 
    "Touche2020"
] # Small datasets.

def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Process model key")
    parser.add_argument(
        "model_key", 
        help="The key for the model", 
        type=str, 
        choices=MODEL_FOLDER_DICT.keys()
    )
    return parser.parse_args()

# def batched_cos_sim(a: torch.Tensor, b: torch.Tensor, batch_size: int) -> torch.Tensor:
#     cos_sims = []
#     i = 0
#     while i < len(b):
#         cos_sims.append(
#             cos_sim(a, b[i: i+batch_size])
#         )
#         i += batch_size
#     return torch.cat(cos_sims, dim=0)

def main():
    args = parse_args()
    model_folder = MODEL_FOLDER_DICT[args.model_key]
    args_str = ARGS_STR_DICT[args.model_key]
    trainer, (model_args, data_args, training_args) = analyze_utils.load_trainer_from_checkpoint_and_args(
        model_folder=model_folder,
        args_str=args_str,
        load_from_checkpoint=True,
        return_args=True
    )
    random.seed(42)
    mteb_encoder = DenseEncoder(
        model_name_or_path=trainer.model.config.embedder,
        # encoder=trainer.model,
        encoder=trainer.model.second_stage_model,
        max_seq_length=trainer.model.config.max_seq_length,
        query_prefix="search_query: " if data_args.use_prefix else "",
        document_prefix="search_document: " if data_args.use_prefix else "",
    )

    n_ensemble = 1
    for task_idx, task in enumerate(TASK_LIST_RETRIEVAL):
        print(f"Beginning {task} ({task_idx+1} / {len(TASK_LIST_RETRIEVAL)})")
        evaluation = MTEB(tasks=[task], task_langs=["en"])
        ##################################################
        evaluation.tasks[0].load_data()
        all_documents = list(evaluation.tasks[0].corpus["test"].values())
        all_dataset_embeddings = []
        corpus_documents = random.choices(all_documents, k=256)
        corpus_documents = [
            document.get("title", "") + document.get("text", "")
            for document in corpus_documents
        ]
        mteb_encoder.encoder = trainer.model.first_stage_model
        dataset_embeddings = torch.tensor(
            mteb_encoder.encode_corpus(corpus_documents)
        )
        dataset_embeddings /= n_ensemble
        mteb_encoder._model_is_on_device = False
        mteb_encoder.encoder = trainer.model.second_stage_model
        ##################################################
        mteb_encoder.encoder.forward = functools.partial(
            mteb_encoder.encoder.forward,
            dataset_embeddings=dataset_embeddings
        )
        results = evaluation.run(
            mteb_encoder, 
            output_folder=os.path.join("results_mteb", args.model_key),
            batch_size=256, 
            corpus_chunk_size=100_000,
            verbosity=2
        )
        print(task)
        print("\t", results)
        print()
    

if __name__ == '__main__':
    main()
