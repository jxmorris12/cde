import argparse
import collections
import functools
import os
import random

import torch

from spider.lib import cluster_dataset
from spider.lib.embed import DenseEncoder
from spider.lib.model_configs import MODEL_FOLDER_DICT
from spider.lib.utils import analyze_utils

from mteb import MTEB
from mteb import HFDataLoader
# from spider.lib.eval.mteb import MTEB

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


def main():
    args = parse_args()
    model_folder = MODEL_FOLDER_DICT[args.model_key]
    trainer, (model_args, data_args, training_args) = analyze_utils.load_trainer_from_checkpoint_and_args(
        model_folder=model_folder,
        load_from_checkpoint=True,
        return_args=True
    )
    trainer.model.eval()
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
        cluster_matches = cluster_dataset(
            dataset=corpus_ds,
            model="gtr_base",
            query_key="text",
            document_key="text",
            query_to_doc=True,
            cluster_size=256,
        )
        cluster_assignments = collections.defaultdict(list)
        for k, v in cluster_matches.items():
            cluster_assignments[v[0]].append(k)
        ##################################################
        all_documents = list(corpus.values())
        all_dataset_embeddings = []
        random.seed(52)
        ##################################################
        corpus_document_ids = []
        while len(corpus_document_ids) < 256:
            cluster_idx = random.choice(list(cluster_assignments.keys()))
            corpus_document_ids.extend(cluster_assignments[cluster_idx])
        corpus_document_ids = random.shuffle(corpus_document_ids)[:256]
        corpus_documents = [corpus_ds[id] for id in corpus_document_ids]
        ##################################################
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
            batch_size=1024, 
            corpus_chunk_size=500_000,
            verbosity=2,
            eval_splits=[split]
        )
        print(task)
        print("\t", results)
        print()
    

if __name__ == '__main__':
    main()
