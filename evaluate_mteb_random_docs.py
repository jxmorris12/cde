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

from mteb import MTEB


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
# TASK_LIST = TASK_LIST_RETRIEVAL
# TASK_LIST = TASK_LIST_STS
# TASK_LIST = TASK_LIST_CLUSTERING
# TASK_LIST = TASK_LIST_PAIR_CLASSIFICATION
# TASK_LIST = TASK_LIST_RERANKING
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
    mteb_encoder = DenseEncoder(
        model_name_or_path=trainer.model.config.embedder,
        encoder=trainer.model.second_stage_model,
        max_seq_length=trainer.model.config.max_seq_length,
        query_prefix="",     # Set later
        document_prefix="",  # Set later
        normalize_embeds=NORMALIZE_EMBEDS,
        default_doc_prefix=True,
    )

    corpus_documents = open("text_data/random_docs.txt", "r").readlines()
    corpus_documents = random.choices(corpus_documents, k=trainer.model.config.transductive_corpus_size)
    dataset_inputs = mteb_encoder.tokenizer(
        corpus_documents,
        return_tensors="pt",
        max_length=trainer.model.config.max_seq_length,
        padding=True,
        truncation=True,
    ).to(training_args.device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        dataset_embeddings = trainer.model.first_stage_model(
            **dataset_inputs
        )
    hidden_dim = dataset_embeddings.shape[-1]
    dataset_embeddings = dataset_embeddings.reshape((1, -1, hidden_dim)) # flatten for multiple contextual tokens
    dataset_embeddings = dataset_embeddings.to(torch.float32).cpu().numpy()

    random.Random(time.time()).shuffle(TASK_LIST)
    for task_idx, task in enumerate(TASK_LIST):
        prefixes = task2prefix[task]
        mteb_encoder.document_prefix = (prefixes["document"] + ": ") if data_args.use_prefix else ""
        mteb_encoder.query_prefix = (prefixes["query"] + ": ") if data_args.use_prefix else ""
        mteb_encoder.normalize_embeds = "Clustering" in task
        print(f"[{task}] Set prefixes to {mteb_encoder.query_prefix} and {mteb_encoder.document_prefix}")

        print(f"Beginning {task} ({task_idx+1} / {len(TASK_LIST)})")
        evaluation = MTEB(
            tasks=[task], 
            task_langs=["en"],
            embedder_rerank="sentence-transformers/gtr-t5-base",
        )
        split = "dev" if task == "MSMARCO" else "test"
        ##################################################
        trainer.model.first_stage_model.cuda()
        ##################################################
        mteb_encoder.model_kwargs = {
            "dataset_embeddings": dataset_embeddings,
            "null_dataset_embedding": False,
            # ""
        }
        # breakpoint()
        results = evaluation.run(
            mteb_encoder, 
            output_folder=os.path.join("results_mteb", "random_documents", args.model_key),
            batch_size=512, 
            # batch_size=128, 
            corpus_chunk_size=500_000,
            verbosity=2,
            eval_splits=[split]
        )
        print(task)
        print("\t", results)
        if len(results):
            results_dict = results[0].to_dict()["scores"][split][0]
            try:
                print("main_score =>", results_dict["main_score"])
            except KeyError:
                print(results_dict)
                continue
        print()
    

if __name__ == '__main__':
    main()
