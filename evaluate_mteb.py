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
TASK_LIST = ["ArguAna"]

def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Process model key")
    parser.add_argument(
        "model_key", 
        help="The key for the model", 
        type=str, 
        choices=MODEL_FOLDER_DICT.keys()
    )
    parser.add_argument(
        "--batch_size", 
        help="Batch size for evaluation", 
        type=int, 
        default=512,
    )
    return parser.parse_args()


NORMALIZE_EMBEDS = False
def main():
    args = parse_args()
    model_folder = MODEL_FOLDER_DICT[args.model_key]
    model, (model_args, data_args, training_args) = analyze_utils.load_trainer_from_checkpoint_and_args(
        model_folder=model_folder,
        load_from_checkpoint=True,
        return_args=True,
        load_entire_trainer=False
    )
    model.eval()
    
    gc.collect()
    torch.cuda.empty_cache()

    datasets.enable_caching()
    mteb_encoder = DenseEncoder(
        model_name_or_path=model.config.embedder,
        encoder=model.second_stage_model,
        max_seq_length=model.config.max_seq_length,
        query_prefix="",     # Set later
        document_prefix="",  # Set later
        normalize_embeds=NORMALIZE_EMBEDS,
        default_doc_prefix=True,
    )

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
        evaluation.tasks[0].load_data()
        if hasattr(evaluation.tasks[0], 'corpus') and split in evaluation.tasks[0].corpus:
            corpus = evaluation.tasks[0].corpus[split]
            documents = random.choices(list(corpus.values()), k=model.config.transductive_corpus_size)
            corpus_documents = [
                mteb_encoder.document_prefix + '{} {}'.format(doc.get('title', ''), doc['text']).strip() 
                for doc in documents
            ]
        elif hasattr(evaluation.tasks[0], 'dataset'):
            if isinstance(evaluation.tasks[0].dataset, datasets.Dataset):
                dataset = evaluation.tasks[0].dataset
            elif split in evaluation.tasks[0].dataset:
                dataset = evaluation.tasks[0].dataset[split]
            else:
                dataset = next(iter(evaluation.tasks[0].dataset.values()))
                if split in dataset:
                    dataset = dataset[split]
                else:
                    dataset = next(iter(dataset.values()))
            column_names = set(dataset.column_names)
            if {"sentence1", "sentence2"} <= column_names:
                documents = dataset["sentence1"] + dataset["sentence2"]
            elif {"sentences"} <= column_names:
                documents = dataset["sentences"]
            elif {"text"} <= column_names:
                documents = dataset["text"]
            elif {"query", "positive"} <= column_names:
                documents = dataset["positive"]
            else:
                raise ValueError(f"No corpus or dataset - got {column_names}")
            if isinstance(documents[0], list):
                documents = [sentence for doc in documents for sentence in doc]
            documents = random.choices(documents, k=model.config.transductive_corpus_size)
            corpus_documents = [
                mteb_encoder.document_prefix + doc
                for doc in documents
            ]
        else:
            raise ValueError("No corpus or dataset available")
        ##################################################
        print(f"Got {len(corpus_documents)} documents...")
        assert len(corpus_documents) == model.config.transductive_corpus_size
        print(corpus_documents[2])
        model.first_stage_model.cuda()
        dataset_inputs = mteb_encoder.tokenizer(
            corpus_documents,
            return_tensors="pt",
            max_length=model.config.max_seq_length,
            padding=True,
            truncation=True,
        ).to(training_args.device)
        ##################################################
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            dataset_embeddings = model.first_stage_model(
                **dataset_inputs
            )
        hidden_dim = dataset_embeddings.shape[-1]
        dataset_embeddings = dataset_embeddings.reshape((1, -1, hidden_dim)) # flatten for multiple contextual tokens
        dataset_embeddings = dataset_embeddings.to(torch.float32).cpu().numpy()
        ##################################################
        mteb_encoder.model_kwargs = {
            "dataset_embeddings": dataset_embeddings,
            "null_dataset_embedding": False,
            # ""
        }
        # breakpoint()
        results = evaluation.run(
            mteb_encoder, 
            output_folder=os.path.join("results_mteb", args.model_key),
            corpus_chunk_size=500_000,
            verbosity=2,
            eval_splits=[split],
            encode_kwargs={"batch_size": args.batch_size, "num_workers": 8 },
            # encode_kwargs={"batch_size": args.batch_size, "num_workers": 0},
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
