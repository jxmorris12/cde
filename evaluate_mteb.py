import argparse
import os

from lib.embed import DenseEncoder
from lib.utils import analyze_utils

from mteb import MTEB


# TODO: Save configs or push to hub. Make this nicer.
ARGS_STR_DICT = {
    # this is my reimplementation of the nomic biencoder. it only trained for around 75%
    # of total training, but with a slightly higher learning rate, but i expect slightly
    # worse MTEB results here.
    # https://wandb.ai/jack-morris/tti-nomic-5/runs/1j3s5rmo/overview?nw=nwuserjxmorris12
    "unsupervised-reimpl": "--per_device_train_batch_size 2048 --per_device_eval_batch_size 2048 --use_wandb 1 --bf16 1 --dataset nomic_unsupervised --sampling_strategy domain --exp_name biencoder-scratch-3--domain --num_train_epochs 3 --learning_rate 2e-5 --embedder nomic-ai/nomic-bert-2048 --dataset_embedder nomic-ai/nomic-bert-2048 --clustering_model gtr_base --clustering_query_to_doc 1 --automatically_deduplicate_documents 1 --automatically_deduplicate_queries 1 --arch biencoder --dataset_info batch --ddp_find_unused_parameters 0 --torch_compile 0 --eval_rerank_topk 1024 --lr_scheduler_type cosine --warmup_steps 5600 --disable_dropout 1 --eval_steps 10000 --max_seq_length 512 --max_batch_size_fits_in_memory 128 --use_gc 1 --logging_steps 20 --train_cluster_size 224 --eval_cluster_size 224 --use_prefix 1 --transductive_corpus_size 1024 --save_steps 4000 --logit_scale 50 --max_eval_batches 2 --adam_beta2 0.95",


    # this is my OG model trained with hard batching for a little over an epoch. if these 
    # eval metrics look OK i'm gonna keep training it for longer and see if it gets better.
    # i tried it (see -hard-batch-1) but accidentally changed the cluster size and messed
    # up the model.
    "unsupervised-hard-batch-0": "--per_device_train_batch_size 2048 --per_device_eval_batch_size 2048 --use_wandb 1 --bf16 1 --dataset nomic_unsupervised --sampling_strategy cluster_within_domain --exp_name biencoder-scratch-3--cluster --num_train_epochs 3 --learning_rate 2e-5 --embedder nomic-ai/nomic-bert-2048 --dataset_embedder nomic-ai/nomic-bert-2048 --clustering_model gtr_base --clustering_query_to_doc 1 --automatically_deduplicate_documents 1 --automatically_deduplicate_queries 1 --arch biencoder --dataset_info batch --ddp_find_unused_parameters 0 --torch_compile 0 --eval_rerank_topk 1024 --lr_scheduler_type cosine --warmup_steps 5600 --disable_dropout 1 --eval_steps 10000 --max_seq_length 512 --max_batch_size_fits_in_memory 128 --use_gc 1 --logging_steps 20 --train_cluster_size 16384 --eval_cluster_size 16384 --use_prefix 1 --transductive_corpus_size 1024 --save_steps 400 --contrastive_temp 50 --max_eval_batches 2 --adam_beta2 0.95",

    # this is my model trained with hard batching. it was trained for about two epochs,
    # maybe a bit less. it's trained longer than the baseline but we're going for SOTA
    # so i think that's allowed in this case. by accident though i trained it with 224-
    # sized clusters after 
    # https://wandb.ai/jack-morris/tti-nomic-5/runs/m44bucij/overview?nw=nwuserjxmorris12
    "unsupervised-hard-batch-1": "--per_device_train_batch_size 2048 --per_device_eval_batch_size 2048 --use_wandb 1 --bf16 1 --dataset nomic_unsupervised --sampling_strategy cluster_within_domain --exp_name biencoder-scratch-4--cluster--continue --num_train_epochs 4 --learning_rate 2e-5 --embedder nomic-ai/nomic-bert-2048 --dataset_embedder nomic-ai/nomic-bert-2048 --clustering_model gtr_base --clustering_query_to_doc 1 --automatically_deduplicate_documents 1 --automatically_deduplicate_queries 1 --arch biencoder --dataset_info batch --ddp_find_unused_parameters 0 --torch_compile 0 --eval_rerank_topk 1024 --lr_scheduler_type inverse_sqrt --warmup_steps 5600 --disable_dropout 1 --eval_steps 4000 --max_seq_length 512 --max_batch_size_fits_in_memory 128 --use_gc 1 --logging_steps 20 --train_cluster_size 224 --eval_cluster_size 224 --use_prefix 1 --transductive_corpus_size 1024 --save_steps 400 --logit_scale 50 --max_eval_batches 4 --adam_beta2 0.95 --exp_name biencoder-scratch-2--cluster-longer --resume_from_checkpoint saves/2024-04-11-biencoder-scratch-3--cluster/checkpoint-15800/",

}

MODEL_FOLDER_DICT = {
    "unsupervised-reimpl": "/home/paperspace/tti3/saves/2024-04-12-biencoder-scratch-3--domain/",
    "unsupervised-hard-batch-0"
    "unsupervised-hard-batch-1": "/home/paperspace/tti3/saves/2024-04-13-biencoder-scratch-2--cluster-longer",
}

assert ARGS_STR_DICT.keys() == MODEL_FOLDER_DICT.keys()

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
    "FEVER",
    "FiQA2018",
    "HotpotQA",
    "MSMARCO",
    "NFCorpus",
    "NQ",
    "QuoraRetrieval",
    "SCIDOCS",
    "SciFact",
    "Touche2020",
    "TRECCOVID",
]


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
        query_prefix="search_query",
        document_prefix="search_document",
    )

    for task_idx, task in enumerate(TASK_LIST_RETRIEVAL):
        print(f"Beginning {task} ({task_idx+1} / {len(TASK_LIST_RETRIEVAL)})")
        evaluation = MTEB(tasks=[task], task_langs=["en"])
        evaluation.run(
            mteb_encoder, 
            output_folder=os.path.join("results_mteb", args.model_key),
            batch_size=1024, 
            corpus_chunk_size=2_000_000
        )
    

if __name__ == '__main__':
    main()
