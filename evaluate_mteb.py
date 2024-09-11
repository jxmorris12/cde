import argparse
import os
import random

import datasets
import torch

from cde.lib import cluster_dataset
from cde.lib.embed import DenseEncoder
from cde.lib.model_configs import MODEL_FOLDER_DICT
from cde.lib.utils import analyze_utils

from mteb import MTEB
from mteb import HFDataLoader

TASK_LIST_RETRIEVAL = [
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
    "TRECCOVID",
    "FEVER",
    "MSMARCO",
]

# TASK_LIST_RETRIEVAL = ["SCIDOCS", "SciFact", "NFCorpus", "TRECCOVID", "Touche2020"] # Small datasets.
# TASK_LIST_RETRIEVAL = ["TRECCOVID"]
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
TASK_LIST_RETRIEVAL = ["ArguAna"]

def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Process model key")
    parser.add_argument(
        "model_key", 
        help="The key for the model", 
        type=str, 
        choices=MODEL_FOLDER_DICT.keys()
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_folder = MODEL_FOLDER_DICT[args.model_key]
    trainer, (model_args, data_args, training_args) = analyze_utils.load_trainer_from_checkpoint_and_args(
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

    for task_idx, task in enumerate(TASK_LIST_RETRIEVAL):
        print(f"Beginning {task} ({task_idx+1} / {len(TASK_LIST_RETRIEVAL)})")
        evaluation = MTEB(
            tasks=[task], 
            task_langs=["en"],
            embedder_rerank="sentence-transformers/gtr-t5-base"
        )
        split = "dev" if task == "MSMARCO" else "test"
        ##################################################
        evaluation.tasks[0].load_data()
        corpus = evaluation.tasks[0].corpus["test"]
        dataset_path = evaluation.tasks[0].metadata_dict["dataset"]["path"]
        hf_repo_qrels = (
            dataset_path + "-qrels" if "clarin-knext" in dataset_path else None
        )
        corpus_ds, _q, _qrels = HFDataLoader(
            hf_repo=dataset_path,
            hf_repo_qrels=hf_repo_qrels,
            streaming=False,
            keep_in_memory=False,
        ).load(split=split)

        ##################################################
        random.seed(52)
        corpus_documents = random.choices(list(corpus.values()), k=trainer.model.config.transductive_corpus_size)
        ##################################################
        corpus_documents = [
            mteb_encoder.document_prefix + '{} {}'.format(doc.get('title', ''), doc['text']).strip() 
            for doc in corpus_documents
        ]
        print(corpus_documents[2])
        trainer.model.first_stage_model.cuda()
        dataset_inputs = mteb_encoder.tokenizer(
            corpus_documents,
            return_tensors="pt",s
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
            corpus_chunk_size=500_000,
            verbosity=2,
            eval_splits=[split]
        )
        print(task)
        print("\t", results)
        if len(results):
            print("NDCG@10 =>", results[0].to_dict()['scores'][split][0]['ndcg_at_10'])
        print()
    

if __name__ == '__main__':
    main()
