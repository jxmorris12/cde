import argparse
import functools
import os

import torch

from spider.lib.embed import DenseEncoder
from spider.lib.model_configs import MODEL_FOLDER_DICT, ARGS_STR_DICT
from spider.lib.utils import analyze_utils

from mteb import MTEB
from mteb.evaluation.evaluators import cos_sim

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

# TASK_LIST_RETRIEVAL = ["SCIDOCS", "SciFact", "NFCorpus", "TRECCOVID", "Touche2020"] # Small datasets.

# TASK_LIST_RETRIEVAL = ["QuoraRetrieval"]


# TODO: Support two-stage models.

# class TwoStageModel():
#     def __init__(self, first_stage_model, second_stage_model):
#         self.first
#     def encode_queries(self, sentences: list[str], **kwargs) -> torch.Tensor:
#         ??
#     def encode_corpus(self, sentences: list[str], **kwargs) -> torch.Tensor:
#         """
#         Returns a list of embeddings for the given sentences.
#         Args:
#             sentences: List of sentences to encode
#         Returns:
#             List of embeddings for the given sentences
#         """
#         pass

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
    trainer = analyze_utils.load_trainer_from_checkpoint_and_args(
        model_folder=model_folder,
        args_str=args_str,
    )
    mteb_encoder = DenseEncoder(
        model_name_or_path=trainer.model.config.embedder,
        max_seq_length=trainer.model.config.max_seq_length,
        encoder=trainer.model,
        query_prefix="search_query: ",
        document_prefix="search_document: ",
    )

    for task_idx, task in enumerate(TASK_LIST_RETRIEVAL):
        print(f"Beginning {task} ({task_idx+1} / {len(TASK_LIST_RETRIEVAL)})")
        evaluation = MTEB(tasks=[task], task_langs=["en"])
        #evaluation.score_functions["cos_sim"] = functools.partial(
        #    batched_cos_sim, batch_size=50_000
        #)
        results = evaluation.run(
            mteb_encoder, 
            output_folder=os.path.join("results_mteb", args.model_key),
            batch_size=1024, 
            corpus_chunk_size=100_000,
            verbosity=2
        )
        print(task)
        print("\t", results)
        print()
    

if __name__ == '__main__':
    main()
