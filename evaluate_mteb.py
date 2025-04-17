import argparse
import gc
import os
import random
import time

import datasets
import torch
import transformers

# from cde.lib import cluster_dataset
from cde.lib.embed import DenseEncoder
from cde.lib.eval.mteb import (
    TASK_LIST_STS, 
    TASK_LIST, 
    task2prefix_short, 
    task2prefix_long, 
    TASK_LIST_RETRIEVAL, 
    TASK_LIST_CLUSTERING, 
    TASK_LIST_PAIR_CLASSIFICATION, 
    TASK_LIST_RERANKING
)
from cde.lib.model_configs import MODEL_FOLDER_DICT
from cde.lib.utils import analyze_utils

from mteb import MTEB


os.environ['OPENBLAS_NUM_THREADS'] = '16'

TASKS_BY_CATEGORY = {
    "retrieval": TASK_LIST_RETRIEVAL,
    "retrieval_tiny": ["ArguAna", "NFCorpus"],
    "sts": TASK_LIST_STS,
    "clustering": TASK_LIST_CLUSTERING,
    "pair_classification": TASK_LIST_PAIR_CLASSIFICATION,
    "reranking": TASK_LIST_RERANKING,
    "all": TASK_LIST,
}

def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Process model key")
    parser.add_argument(
        "model_key", 
        help="The key for the model", 
        type=str, 
        # choices=MODEL_FOLDER_DICT.keys()
    )
    parser.add_argument(
        "--batch_size", 
        help="Batch size for evaluation", 
        type=int, 
        default=512,
    )
    parser.add_argument(
        "--tasks",
        choices=TASKS_BY_CATEGORY.keys(),
        default="all",

    )
    return parser.parse_args()


def run_eval_contextual(args, evaluation, split, mteb_encoder, model, training_args, first_stage_tokenizer, corpus_documents):
    model.first_stage_model.cuda()

    i = 0
    dataset_embeddings = []
    while i < len(corpus_documents):
        dataset_inputs = first_stage_tokenizer(
            corpus_documents[i:i+args.batch_size],
            return_tensors="pt",
            max_length=model.config.max_seq_length,
            padding=True,
            truncation=True,
        ).to(training_args.device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            batch_dataset_embeddings = model.first_stage_model(
                **dataset_inputs
            )
        dataset_embeddings.append(batch_dataset_embeddings)
        i += args.batch_size
    dataset_embeddings = torch.cat(dataset_embeddings, dim=0).to(training_args.device)
    hidden_dim = dataset_embeddings.shape[-1]
    dataset_embeddings = dataset_embeddings.reshape((1, -1, hidden_dim)) # flatten for multiple contextual tokens
    dataset_embeddings = dataset_embeddings.to(torch.float32).cpu().numpy()

    ##################################################
    mteb_encoder.model_kwargs = {
        "dataset_embeddings": dataset_embeddings,
        "null_dataset_embedding": False,
        # ""
    }
    sanitized_model_key = args.model_key.replace("/", "_").strip("_")
    print("output_folder", os.path.join("results_mteb", sanitized_model_key))
    return evaluation.run(
        mteb_encoder, 
        output_folder=os.path.join("results_mteb", sanitized_model_key),
        corpus_chunk_size=500_000,
        verbosity=2,
        eval_splits=[split],
        encode_kwargs={"batch_size": args.batch_size},
    )


def run_eval_biencoder(args, evaluation, split, mteb_encoder):
    sanitized_model_key = args.model_key.replace("/", "_").strip("_")
    return evaluation.run(
        mteb_encoder, 
        output_folder=os.path.join("results_mteb", sanitized_model_key),
        corpus_chunk_size=500_000,
        verbosity=2,
        eval_splits=[split],
        encode_kwargs={"batch_size": args.batch_size },
    )


def main():
    args = parse_args()
    model_folder = MODEL_FOLDER_DICT.get(args.model_key, args.model_key)
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

    if hasattr(model.config, "dataset_backbone") and model.config.dataset_backbone:
        model_name_or_path = model.config.dataset_backbone
    else:
        model_name_or_path = model.config.embedder
    
    mteb_encoder = DenseEncoder(
        model_name_or_path=model_name_or_path,
        encoder=model.second_stage_model if hasattr(model, "second_stage_model") else model,
        max_seq_length=model.config.max_seq_length,
        query_prefix="",        # Set later; depends on task.
        document_prefix="",     # Set later; depends on task.
        normalize_embeds=False, # Set later; depends on task.
        default_doc_prefix=True,
    )
    first_stage_tokenizer = transformers.AutoTokenizer.from_pretrained(model.config.embedder)
    second_stage_tokenizer = transformers.AutoTokenizer.from_pretrained(vars(model.config).get("dataset_backbone") or model.config.embedder)

    tasks = TASKS_BY_CATEGORY[args.tasks]

    random.Random(time.time()).shuffle(tasks)
    for task_idx, task in enumerate(tasks):
        document_prefix = ""
        query_prefix = ""
        if data_args.use_prefix:
            short_prefixes = task2prefix_short[task]
            is_symmetric = (short_prefixes["query"] == short_prefixes["document"])
            if data_args.use_short_prefix:
                document_prefix = (short_prefixes["document"] + ": ") if data_args.use_prefix else ""
                query_prefix = (short_prefixes["query"] + ": ") if data_args.use_prefix else ""
            else:
                query_prefix = task2prefix_long[task] + second_stage_tokenizer.bos_token + " "
                document_prefix = query_prefix if is_symmetric else ""
                
        mteb_encoder.document_prefix = document_prefix
        mteb_encoder.query_prefix = query_prefix

        print(f"Set prefixes to {mteb_encoder.query_prefix} and {mteb_encoder.document_prefix}")
        mteb_encoder.normalize_embeds = "Clustering" in task
        print(f"[{task}] Set prefixes to {mteb_encoder.query_prefix} and {mteb_encoder.document_prefix}")

        print(f"Beginning {task} ({task_idx+1} / {len(tasks)})")
        evaluation = MTEB(
            tasks=[task], 
            task_langs=["en"],
        )
        split = "dev" if task == "MSMARCO" else "test"
        ####################################################################################################
        evaluation.tasks[0].load_data()
        if hasattr(evaluation.tasks[0], 'corpus') and split in evaluation.tasks[0].corpus:
            corpus = evaluation.tasks[0].corpus[split]
            documents = random.choices(list(corpus.values()), k=model.config.transductive_corpus_size)
            corpus_documents = [
                mteb_encoder.document_prefix + doc
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

        if hasattr(model, "first_stage_model"):
            print(f"[evaluate_mteb] Running contextual evaluation with seq_length={model.config.max_seq_length} and transductive_corpus_size={model.config.transductive_corpus_size}")
            results = run_eval_contextual(
                args=args,
                evaluation=evaluation, 
                split=split, 
                mteb_encoder=mteb_encoder, 
                model=model,
                training_args=training_args, 
                first_stage_tokenizer=first_stage_tokenizer, 
                corpus_documents=corpus_documents
            )
        else:
            print("[evaluate_mteb] Running biencoder evaluation")
            results = run_eval_biencoder(
                args=args, 
                evaluation=evaluation, 
                split=split, 
                mteb_encoder=mteb_encoder
            )
        
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
