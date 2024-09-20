import argparse
import gc
import os
import random
import time

import datasets
import torch

# from cde.lib import cluster_dataset
from cde.lib.embed import DenseEncoder
from cde.lib.model_configs import MODEL_FOLDER_DICT
from cde.lib.utils import analyze_utils

# from mteb import MTEB
from cde.lib.eval.mteb import MTEB


os.environ['OPENBLAS_NUM_THREADS'] = '16'

TASK_LIST_CLASSIFICATION = [
    "AmazonCounterfactualClassification",
    "AmazonPolarityClassification",
    "AmazonReviewsClassification",
    "Banking77Classification",
    "EmotionClassification",
    "ImdbClassification",
    "MassiveIntentClassification",
    "MassiveScenarioClassification",
    "MTOPDomainClassification",
    "MTOPIntentClassification",
    "ToxicConversationsClassification",
    "TweetSentimentExtractionClassification",
]

TASK_LIST_CLUSTERING = [
    "ArxivClusteringP2P",
    "ArxivClusteringS2S",
    "BiorxivClusteringP2P",
    "BiorxivClusteringS2S",
    "MedrxivClusteringP2P",
    "MedrxivClusteringS2S",
    "RedditClustering",
    "RedditClusteringP2P",
    "StackExchangeClustering",
    "StackExchangeClusteringP2P",
    "TwentyNewsgroupsClustering",
]

TASK_LIST_PAIR_CLASSIFICATION = [
    "SprintDuplicateQuestions",
    "TwitterSemEval2015",
    "TwitterURLCorpus",
]

TASK_LIST_RERANKING = [
    "AskUbuntuDupQuestions",
    "MindSmallReranking",
    "SciDocsRR",
    "StackOverflowDupQuestions",
]

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
# TASK_LIST_RETRIEVAL = ["ArguAna"]

TASK_LIST_STS = [
    "BIOSSES",
    "SICK-R",
    "STS12",
    "STS13",
    "STS14",
    "STS15",
    "STS16",
    "STS17",
    "STS22",
    "STSBenchmark",
    "SummEval",
]

task2prefix = {}
for task in TASK_LIST_CLASSIFICATION:
    task2prefix[task] = {"query": "classification", "document": "classification"}

for task in TASK_LIST_CLUSTERING:
    task2prefix[task] = {"query": "clustering", "document": "clustering"}

for task in TASK_LIST_PAIR_CLASSIFICATION:
    task2prefix[task] = {"query": "classification", "document": "classification"}

for task in TASK_LIST_RERANKING:
    task2prefix[task] = {"query": "classification", "document": "classification"}

for task in TASK_LIST_RETRIEVAL:
    task2prefix[task] = {"query": "search_query", "document": "search_document"}

for task in TASK_LIST_STS:
    task2prefix[task] = {"query": "classification", "document": "classification"}

task2prefix["QuoraRetrieval"] = {"query": "search_query", "document": "search_query"}


TASK_LIST = (
    TASK_LIST_CLASSIFICATION
    + TASK_LIST_CLUSTERING
    + TASK_LIST_PAIR_CLASSIFICATION
    + TASK_LIST_RERANKING
    + TASK_LIST_RETRIEVAL
    + TASK_LIST_STS
)
# TASK_LIST = TASK_LIST_CLASSIFICATION
# TASK_LIST = TASK_LIST_RETRIEVAL
# TASK_LIST = TASK_LIST_STS
# TASK_LIST = TASK_LIST_CLUSTERING
# TASK_LIST = TASK_LIST_PAIR_CLASSIFICATION
TASK_LIST = TASK_LIST_RERANKING
# TASK_LIST = ["Touche2020"]

def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Process model key")
    parser.add_argument(
        "model_key", 
        help="The key for the model", 
        type=str, 
        choices=MODEL_FOLDER_DICT.keys()
    )
    return parser.parse_args()


NORMALIZE_EMBEDS = False
def main():
    args = parse_args()
    model_folder = MODEL_FOLDER_DICT[args.model_key]
    trainer, (model_args, data_args, training_args) = analyze_utils.load_trainer_from_checkpoint_and_args(
        model_folder=model_folder,
        load_from_checkpoint=True,
        return_args=True
    )
    trainer.model.eval()
    trainer._hn_filter_model = None
    
    gc.collect()
    torch.cuda.empty_cache()

    datasets.enable_caching()
    first_stage_encoder = DenseEncoder(
        model_name_or_path=trainer.model.config.embedder,
        encoder=trainer.model.first_stage_model,
        max_seq_length=trainer.model.config.max_seq_length,
        query_prefix="search_query: " if data_args.use_prefix else "",
        document_prefix="search_document: " if data_args.use_prefix else "",
        normalize_embeds=False,
        default_doc_prefix=True,
    )

    mteb_encoder = DenseEncoder(
        model_name_or_path=trainer.model.config.embedder,
        encoder=trainer.model.second_stage_model,
        max_seq_length=trainer.model.config.max_seq_length,
        query_prefix="search_query: " if data_args.use_prefix else "",
        document_prefix="search_document: " if data_args.use_prefix else "",
        normalize_embeds=NORMALIZE_EMBEDS,
        default_doc_prefix=True,
    )
    # hacky way to get around editing all the files in MTEB
    mteb_encoder.first_stage_encoder = first_stage_encoder
    mteb_encoder.corpus_size = trainer.model.config.transductive_corpus_size

    random.Random(time.time()).shuffle(TASK_LIST)
    for task_idx, task in enumerate(TASK_LIST):
        prefixes = task2prefix[task]
        mteb_encoder.document_prefix = (prefixes["document"] + ": ") if data_args.use_prefix else ""
        mteb_encoder.query_prefix = (prefixes["query"] + ": ") if data_args.use_prefix else ""
        print(f"Set prefixes to {mteb_encoder.query_prefix} and {mteb_encoder.document_prefix}")

        print(f"Beginning {task} ({task_idx+1} / {len(TASK_LIST)})")
        evaluation = MTEB(
            tasks=[task], 
            task_langs=["en"],
        )
        split = "dev" if task == "MSMARCO" else "test"
        ##################################################
        # breakpoint()
        results = evaluation.run(
            mteb_encoder, 
            output_folder=os.path.join("results_mteb", "local", "norm", args.model_key) if NORMALIZE_EMBEDS else os.path.join("results_mteb", "local", args.model_key),
            batch_size=512, 
            # batch_size=128, 
            corpus_chunk_size=500_000,
            verbosity=2,
            eval_splits=[split]
        )
        print(task)
        print("\t", results)
        if len(results):
            results_dict = next(iter(results.values()))[split]
            # results_dict = results[0].to_dict()["scores"][split][0]
            try:
                print("main_score =>", results_dict["main_score"])
            except KeyError:
                print(results_dict)
                continue
        print()
    

if __name__ == '__main__':
    main()
