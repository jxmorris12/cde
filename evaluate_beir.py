import argparse
import glob
import json
import os
import pandas as pd
import tqdm

from spider.lib.misc import md5_hash_kwargs
from spider.lib.utils import analyze_utils


# can run with torchrun, for example:
#   > torchrun --nproc_per_node 8 evaluate_beir.py \
#               biencoder-scratch-02-cluster224
# 

# TODO: Save configs or push to hub. Make this nicer.
ARGS_STR_DICT = {
    # this is my reimplementation of the nomic biencoder. it only trained for around 75%
    # of total training, but with a slightly higher learning rate, but i expect slightly
    # worse MTEB results here.
    # https://wandb.ai/jack-morris/tti-nomic-5/runs/1j3s5rmo/overview?nw=nwuserjxmorris12
    "unsupervised-reimpl": "--per_device_train_batch_size 2048 --per_device_eval_batch_size 2048 --use_wandb 1 --bf16 1 --dataset nomic_unsupervised --sampling_strategy domain --exp_name biencoder-scratch-3--domain --num_train_epochs 3 --learning_rate 2e-5 --embedder nomic-ai/nomic-bert-2048 --clustering_model gtr_base --clustering_query_to_doc 1 --automatically_deduplicate_documents 1 --automatically_deduplicate_queries 1 --arch biencoder --dataset_info batch --ddp_find_unused_parameters 0 --torch_compile 0 --eval_rerank_topk 1024 --lr_scheduler_type cosine --warmup_steps 5600 --disable_dropout 1 --eval_steps 10000 --max_seq_length 512 --max_batch_size_fits_in_memory 128 --use_gc 1 --logging_steps 20 --train_cluster_size 224 --eval_cluster_size 224 --use_prefix 1 --transductive_corpus_size 1024 --save_steps 4000 --logit_scale 50 --max_eval_batches 2 --adam_beta2 0.95",

    # this is my model trained with hard batching. it was trained for about two epochs,
    # maybe a bit less. it's trained longer than the baseline but we're going for SOTA
    # so i think that's allowed in this case. by accident though i trained it with 224-
    # sized clusters after 
    # https://wandb.ai/jack-morris/tti-nomic-5/runs/m44bucij/overview?nw=nwuserjxmorris12
    # "biencoder-scratch-01-cluster16k": "--per_device_train_batch_size 2048 --per_device_eval_batch_size 2048 --use_wandb 1 --bf16 1 --dataset nomic_unsupervised --sampling_strategy cluster_within_domain --num_train_epochs 1 --learning_rate 2e-5 --embedder nomic-ai/nomic-bert-2048 --clustering_model gtr_base --clustering_query_to_doc 1 --automatically_deduplicate_documents 1 --automatically_deduplicate_queries 1 --arch biencoder --dataset_info batch --ddp_find_unused_parameters 0 --torch_compile 0 --eval_rerank_topk 2048 --lr_scheduler_type inverse_sqrt --warmup_steps 5600 --disable_dropout 1 --eval_steps 10000 --max_seq_length 512 --max_batch_size_fits_in_memory 128 --use_gc 1 --logging_steps 20 --train_cluster_size 16384 --eval_cluster_size 224 --use_prefix 1 --transductive_corpus_size 1024 --save_steps 1000 --logit_scale 50 --max_eval_batches 4 --exp_name 2024-04-14-biencoder-scratch-01-cluster16k",
    # "biencoder-scratch-01-cluster224": "--per_device_train_batch_size 2048 --per_device_eval_batch_size 2048 --use_wandb 1 --bf16 1 --dataset nomic_unsupervised --sampling_strategy cluster_within_domain --num_train_epochs 1 --learning_rate 2e-5 --embedder nomic-ai/nomic-bert-2048 --clustering_model gtr_base --clustering_query_to_doc 1 --automatically_deduplicate_documents 1 --automatically_deduplicate_queries 1 --arch biencoder --dataset_info batch --ddp_find_unused_parameters 0 --torch_compile 0 --eval_rerank_topk 2048 --lr_scheduler_type inverse_sqrt --warmup_steps 5600 --disable_dropout 1 --eval_steps 10000 --max_seq_length 512 --max_batch_size_fits_in_memory 128 --use_gc 1 --logging_steps 20 --train_cluster_size 224 --eval_cluster_size 224 --use_prefix 1 --transductive_corpus_size 1024 --save_steps 1000 --logit_scale 50 --max_eval_batches 4 --exp_name 2024-04-14-biencoder-scratch-01-cluster224",
    "biencoder-scratch-02-cluster224": "--per_device_train_batch_size 224 --per_device_eval_batch_size 224 --use_wandb 1 --bf16 1 --dataset nomic_unsupervised --sampling_strategy cluster_within_domain --num_train_epochs 1 --learning_rate 2e-5 --embedder nomic-ai/nomic-bert-2048 --clustering_model gtr_base --clustering_query_to_doc 1 --automatically_deduplicate_documents 1 --automatically_deduplicate_queries 1 --arch biencoder --dataset_info batch --ddp_find_unused_parameters 0 --eval_rerank_topk 224 --lr_scheduler_type constant_with_warmup --warmup_steps 5600 --disable_dropout 1 --eval_steps 1200000000000 --max_seq_length 512 --use_gc 1 --logging_steps 200 --train_cluster_size 224 --eval_cluster_size 224 --use_prefix 1 --transductive_corpus_size 224 --save_steps 16000 --logit_scale 50 --max_eval_batches 4 --exp_name 2024-04-17-biencoder-pretrain-16 --ddp_share_negatives_between_gpus 0 --max_batch_size_fits_in_memory 128",
    "transductive-scratch-02-cluster224": "--per_device_train_batch_size 224 --per_device_eval_batch_size 224 --use_wandb 1 --bf16 1 --dataset nomic_unsupervised --sampling_strategy cluster_within_domain --num_train_epochs 1 --learning_rate 2e-5 --embedder nomic-ai/nomic-bert-2048 --clustering_model gtr_base --clustering_query_to_doc 1 --automatically_deduplicate_documents 1 --automatically_deduplicate_queries 1 --arch transductive --dataset_info batch --ddp_find_unused_parameters 0 --eval_rerank_topk 224 --lr_scheduler_type constant_with_warmup --warmup_steps 5600 --disable_dropout 1 --eval_steps 1200000000000 --max_seq_length 512 --use_gc 1 --logging_steps 200 --train_cluster_size 224 --eval_cluster_size 224 --use_prefix 1 --transductive_corpus_size 224 --save_steps 16000 --logit_scale 50 --max_eval_batches 4 --exp_name 2024-04-17-transductive-pretrain-16 --ddp_share_negatives_between_gpus 0 --max_batch_size_fits_in_memory 64",

    "biencoder-16cluster-4kbatch-1epoch": "--per_device_train_batch_size 512 --per_device_eval_batch_size 512 --use_wandb 1 --bf16 1 --dataset nomic_unsupervised --sampling_strategy cluster_within_domain --num_train_epochs 2 --learning_rate 2e-5 --embedder nomic-ai/nomic-bert-2048 --clustering_model gtr_base --clustering_query_to_doc 1 --automatically_deduplicate_documents 1 --automatically_deduplicate_queries 1 --arch biencoder --dataset_info batch --ddp_find_unused_parameters 0 --eval_rerank_topk 512 --lr_scheduler_type inverse_sqrt --warmup_steps 5600 --disable_dropout 1 --eval_steps 12000 --max_seq_length 512 --max_batch_size_fits_in_memory 128 --use_gc 1 --logging_steps 20 --train_cluster_size 16384 --eval_cluster_size 224 --use_prefix 1 --transductive_corpus_size 512 --save_steps 1000 --logit_scale 50 --max_eval_batches 4 --exp_name 2024-04-15-biencoder-scratch-03-cluster16k",
    "biencoder-domain-1": "--per_device_train_batch_size 4096 --per_device_eval_batch_size 512 --use_wandb 1 --bf16 1 --dataset nomic_unsupervised --sampling_strategy domain --num_train_epochs 1 --learning_rate 2e-5 --embedder nomic-ai/nomic-bert-2048 --clustering_model gtr_base --clustering_query_to_doc 1 --automatically_deduplicate_documents 1 --automatically_deduplicate_queries 1 --arch biencoder --dataset_info batch --ddp_find_unused_parameters 0 --eval_rerank_topk 512 --lr_scheduler_type inverse_sqrt --warmup_steps 5600 --disable_dropout 1 --eval_steps 12000 --max_seq_length 512 --max_batch_size_fits_in_memory 256 --use_gc 1 --logging_steps 100 --train_cluster_size 16384 --eval_cluster_size 224 --use_prefix 1 --transductive_corpus_size 512 --save_steps 1000 --logit_scale 50 --max_eval_batches 4 --exp_name 2024-04-19-biencoder-scratch-06-domain --ddp_share_negatives_between_gpus=1",

    "supervised-baseline-domain-1epoch": "--per_device_train_batch_size 256 --per_device_eval_batch_size 256 --bf16 1 --use_wandb 1 --dataset nomic_supervised --sampling_strategy domain --num_train_epochs 1 --learning_rate 2e-5 --embedder nomic-ai/nomic-embed-text-v1-unsupervised --clustering_model gtr_base --clustering_query_to_doc 1 --automatically_deduplicate_documents 1 --automatically_deduplicate_queries 1 --arch biencoder --dataset_info batch --eval_rerank_topk 512 --use_prefix 1 --exp_name baseline-supervised-domain --num_hard_negatives 7 --lr_scheduler_type linear --ddp_share_negatives_between_gpus 0 --use_gc 1 --max_batch_size_fits_in_memory 512 --warmup_steps 400 --logging_steps 40 --train_cluster_size 224",
    "supervised-baseline-cluster224-1epoch": "--per_device_train_batch_size 256 --per_device_eval_batch_size 256 --bf16 1 --use_wandb 1 --dataset nomic_supervised --sampling_strategy cluster_within_domain --num_train_epochs 1 --learning_rate 2e-5 --embedder nomic-ai/nomic-embed-text-v1-unsupervised --clustering_model gtr_base --clustering_query_to_doc 1 --automatically_deduplicate_documents 1 --automatically_deduplicate_queries 1 --arch biencoder --dataset_info batch --eval_rerank_topk 512 --use_prefix 1 --exp_name baseline-supervised-cluster224 --num_hard_negatives 7 --lr_scheduler_type linear --ddp_share_negatives_between_gpus 0 --use_gc 1 --max_batch_size_fits_in_memory 512 --warmup_steps 400 --logging_steps 40 --train_cluster_size 224",
    "supervised-baseline-cluster224-1epoch--nodedup": "--per_device_train_batch_size 256 --per_device_eval_batch_size 256 --bf16 1 --use_wandb 1 --dataset nomic_supervised --sampling_strategy cluster_within_domain --num_train_epochs 1 --learning_rate 2e-5 --embedder nomic-ai/nomic-embed-text-v1-unsupervised --clustering_model gtr_base --clustering_query_to_doc 1 --automatically_deduplicate_documents 0 --automatically_deduplicate_queries 0 --arch biencoder --dataset_info batch --eval_rerank_topk 512 --use_prefix 1 --exp_name baseline-supervised-cluster224--no-dedup --num_hard_negatives 7 --lr_scheduler_type linear --ddp_share_negatives_between_gpus 0 --use_gc 1 --max_batch_size_fits_in_memory 256 --warmup_steps 400 --logging_steps 40 --train_cluster_size 224 --save_steps 1000",
    "supervised-baseline-cluster224-1epoch--fixdedup": "--per_device_train_batch_size 256 --per_device_eval_batch_size 256 --bf16 1 --use_wandb 1 --dataset nomic_supervised --sampling_strategy cluster_within_domain --num_train_epochs 1 --learning_rate 2e-5 --embedder nomic-ai/nomic-embed-text-v1-unsupervised --clustering_model gtr_base --clustering_query_to_doc 1 --automatically_deduplicate_documents 1 --automatically_deduplicate_queries 1 --arch biencoder --dataset_info batch --eval_rerank_topk 512 --use_prefix 1 --exp_name baseline-supervised-cluster224--no-combined-dedup --num_hard_negatives 7 --lr_scheduler_type linear --ddp_share_negatives_between_gpus 0 --use_gc 1 --max_batch_size_fits_in_memory 256 --warmup_steps 400 --logging_steps 40 --train_cluster_size 224 --save_steps 1000",

    "transductive-1epoch-supervised-cluster224": "--per_device_train_batch_size 256 --per_device_eval_batch_size 256 --bf16 1 --use_wandb 1 --dataset nomic_supervised --sampling_strategy cluster_within_domain --num_train_epochs 1 --learning_rate 2e-5 --embedder nomic-ai/nomic-embed-text-v1-unsupervised --clustering_model gtr_base --clustering_query_to_doc 1 --automatically_deduplicate_documents 1 --automatically_deduplicate_queries 1 --arch transductive --dataset_info batch --eval_rerank_topk 512 --use_prefix 1 --exp_name transductive-unsupervised-cluster224-1epoch-supervised-cluster224 --num_hard_negatives 7 --lr_scheduler_type linear --ddp_share_negatives_between_gpus 0 --use_gc 1 --max_batch_size_fits_in_memory 128 --warmup_steps 400 --logging_steps 50 --train_cluster_size 224 --save_steps 1000 --model_state_dict_from_path /data/saves/tti3/backup/2024-04-17-transductive-pretrain-16",
}

MODEL_FOLDER_DICT = {
    # "unsupervised-reimpl": "/home/paperspace/tti3/saves/2024-04-12-biencoder-scratch-3--domain/",
    ################################################################################################
    # "biencoder-scratch-01-cluster224": "/home/paperspace/tti3/saves/2024-04-14-biencoder-scratch-01-cluster224/",
    ################################################################################################
    "biencoder-scratch-02-cluster224": "/data/saves/tti3/2024-04-17-biencoder-pretrain-16",
    "transductive-scratch-02-cluster224": "/data/saves/tti3/backup/2024-04-17-transductive-pretrain-16",
    ################################################################################################
    "biencoder-16cluster-4kbatch-1epoch": "/data/saves/tti3/2024-04-15-biencoder-scratch-03-cluster16k-batch4k--1epoch",
    "biencoder-domain-1": "/data/saves/tti3/2024-04-19-biencoder-scratch-06-domain/",
    ################################################################################################
    # baseline + supervised
    "supervised-baseline-domain-1epoch": "/data/saves/tti3/2024-04-22-baseline-supervised-domain/",
    # baseline + hardbatch
    "supervised-baseline-cluster224-1epoch": "/data/saves/tti3/2024-04-22-baseline-supervised-cluster224/",

    # "supervised-baseline-cluster224-1epoch--nodedup": "/data/saves/tti3/2024-04-22-baseline-supervised-cluster224--no-dedup/",
    # "supervised-baseline-cluster224-1epoch--fixdedup": "/data/saves/tti3/2024-04-22-baseline-supervised-cluster224--no-combined-dedup",
    # transductive + Hard batch
    "transductive-1epoch-supervised-cluster224": "/data/saves/tti3/2024-04-22-transductive-unsupervised-cluster224-1epoch-supervised-cluster224",

    # transductive hard batch + hardbatch
    "hardbatch-supervised-cluster224": "/data/saves/tti3/2024-04-22-transductive-unsupervised-cluster224-1epoch-supervised-cluster224",

    # baseline reimpl + hardbatch
    "unsupervised-reimpl-supervised-cluster224": "/data/saves/tti3/2024-04-22-unsupervised-reimpl-supervised-cluster224--2",
}
# assert ARGS_STR_DICT.keys() <= MODEL_FOLDER_DICT.keys(), f"keys not equal: {ARGS_STR_DICT.keys()} != {MODEL_FOLDER_DICT.keys()}"

beir_dataset_names = [ 
    'arguana',
    'fiqa',
    'msmarco',
    'nfcorpus',
    'nq',
    'quora',
    'scidocs', 
    'scifact',
    'signal1m',
    'trec-covid',
    'trec-news',  
    'webis-touche2020',
]

cwd = os.path.normpath(
    os.path.dirname(os.path.abspath(__file__)),
)
root_save_folder = os.path.join(
    cwd, "results_beir"
)


def setup_eval_cmd_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "model_key", 
        help="The key for the model", 
        type=str, 
        choices=MODEL_FOLDER_DICT.keys()
    )
    parser.add_argument(
        "--top_k", "--k",
        type=int, 
        default=8192,
    )
    parser.add_argument(
        "--batch_size", "--b",
        type=int, 
        default=128,
    )
    parser.add_argument(
        "--total", "--n",
        type=int, 
        default=512,
    )

def evaluate_model(args):
    model_folder = MODEL_FOLDER_DICT[args.model_key]
    args_str = ARGS_STR_DICT.get(args.model_key)

    save_folder = os.path.join(root_save_folder, args.model_key)
    os.makedirs(save_folder, exist_ok=True)
    args_dict = vars(args)
    args_dict["datasets"] = tuple(beir_dataset_names)
    save_hash = md5_hash_kwargs(**args_dict)
    save_path = os.path.join(save_folder, save_hash  + ".json")
    if os.path.exists(save_path):
        print(f"found cached results at {save_path}")
        exit()

    trainer = analyze_utils.load_trainer_from_checkpoint_and_args(
        model_folder=model_folder,
        args_str=args_str,
        beir_dataset_names=beir_dataset_names,
        load_from_checkpoint=True, # Set to false for random predictions
    )
    trainer.model.eval()
    trainer.args.max_batch_size_fits_in_memory = args.batch_size
    trainer.args.eval_rerank_topk = args.top_k
    results_dict = trainer.evaluate_retrieval_datasets(n=args.total)
    results_dict["_args"] = args_dict

    if trainer._is_main_worker:
        breakpoint()
        with open(save_path, "w") as json_file:
            json.dump(results_dict, json_file, indent=4)
        print(f"[rank 0] saved {len(results_dict)} results to {save_path}")

def print_results(args):
    print("printing :D")
    results_jsons = glob.glob(os.path.join(root_save_folder, "*", "*.json"))
    all_jsons = [json.load(open(j, "r")) for j in tqdm.tqdm(results_jsons, desc="Reading *.json", leave=False)]
    # add args to outer dict
    for i in range(len(all_jsons)):
        for k, v in all_jsons[i]["_args"].items():
            all_jsons[i][k] = v
    df = pd.DataFrame(all_jsons)
    df = df[df["top_k"] == 1024].reset_index()
    df = df[df["total"] == 1024].reset_index()
    df = df.set_index("model_key")
    df = df[[c for c in df.columns if c.endswith("NDCG@10")]]
    df.columns = [c.replace("eval_BeIR/", "") for c in df.columns]
    df["NDCG@10__mean"] = df.mean(axis=1)
    pd.set_option('display.max_columns', None)
    print(df)
    breakpoint()


def main():
    print("Running something...")
    parser = argparse.ArgumentParser(description="Run and examine BEIR evaluation results")

    # Creating subparsers
    subparsers = parser.add_subparsers(title="subcommands", dest="command", required=True)

    # Subparser for print command
    print_parser = subparsers.add_parser("print", help="Print results df")
    print_parser.set_defaults(func=print_results)

    # Subparser for run command
    eval_parser = subparsers.add_parser("run", help="Run a task")
    # eval_parser.add_argument("eval", help="Evaluate a model")
    eval_parser.set_defaults(func=evaluate_model)
    setup_eval_cmd_parser(eval_parser)

    args = parser.parse_args()
    args.func(args)

    
if __name__ == '__main__':
    main()
