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

# TASK_LIST_RETRIEVAL = ["SCIDOCS", "SciFact", "NFCorpus", "TRECCOVID", "Touche2020"] # Small datasets.
# TASK_LIST_RETRIEVAL = ["TRECCOVID"]
# TASK_LIST_RETRIEVAL = ["FiQA2018"]


# TASK_LIST_RETRIEVAL = [
#     "ArguAna",
#     "FiQA2018", 
#     "NFCorpus", 
#     "SCIDOCS", 
#     "SciFact", 
#     "Touche2020",
#     "TRECCOVID", 
# ] # Small datasets.


# TASK_LIST_RETRIEVAL = ["QuoraRetrieval"]
# TASK_LIST_RETRIEVAL = ["NFCorpus"]

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
    trainer.model.eval()
    mteb_encoder = DenseEncoder(
        model_name_or_path=trainer.model.config.embedder,
        # encoder=trainer.model,
        encoder=trainer.model.second_stage_model,
        max_seq_length=trainer.model.config.max_seq_length,
        query_prefix="search_query: " if data_args.use_prefix else "",
        document_prefix="search_document: " if data_args.use_prefix else "",
    )

    for task_idx, task in enumerate(TASK_LIST_RETRIEVAL):
        print(f"Beginning {task} ({task_idx+1} / {len(TASK_LIST_RETRIEVAL)})")
        evaluation = MTEB(tasks=[task], task_langs=["en"])
        ##################################################
        evaluation.tasks[0].load_data()
        all_documents = list(evaluation.tasks[0].corpus["test"].values())
        all_dataset_embeddings = []
        random.seed(51)
        corpus_documents = random.choices(all_documents, k=256)
        corpus_documents = [
            mteb_encoder.document_prefix + '{} {}'.format(doc.get('title', ''), doc['text']).strip() 
            for doc in corpus_documents
        ]
        print(corpus_documents[2])
        trainer.model.first_stage_model.cuda()
        dataset_inputs = mteb_encoder.tokenizer(
            corpus_documents,
            return_tensors="pt",
            max_length=trainer.model.config.max_seq_length,
            padding=True,
            truncation=True,
        ).to("cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            dataset_embeddings = trainer.model.first_stage_model(
                **dataset_inputs
            )
        ##################################################
        mteb_encoder.model_kwargs = {
            "dataset_embeddings": dataset_embeddings.float().cpu().numpy(),
            "null_dataset_embedding": False,
        }
        results = evaluation.run(
            mteb_encoder, 
            output_folder=os.path.join("results_mteb", args.model_key),
            batch_size=512, 
            corpus_chunk_size=500_000, # 1_000_000,
            verbosity=2,
        )
        print(task)
        print("\t", results)
        print()
    

if __name__ == '__main__':
    main()
