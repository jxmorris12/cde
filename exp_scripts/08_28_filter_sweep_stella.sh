#!/bin/bash
#SBATCH -A memorization
#SBATCH -q memorization_high
#SBATCH --job-name=precompute_stella_multinode
#SBATCH --output=log_submitit/multinode/precompute_stella_%j.out
#SBATCH --error=log_submitit/multinode/precompute_stella_%j.err
#SBATCH --nodes=5
#SBATCH --ntasks=5
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --time=5-00:00:00
#SBATCH --mem=800G
#SBATCH --requeue


# https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/slurm/sbatch_run.sh
echo "[startup] Listing nodes"
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
echo "[startup] Starting head node"
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip

export LOGLEVEL=INFO
export NCCL_P2P_LEVEL=NVL
export NCCL_DEBUG=INFO
export HF_DATASETS_OFFLINE=1 
export HF_HUB_OFFLINE=1



srun torchrun \
    --nnodes 5 \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint $head_node_ip:29500 \
    --nproc_per_node 8 finetune.py --per_device_train_batch_size 1024 --per_device_eval_batch_size 256 --use_wandb 1 --dataset nomic_unsupervised --sampling_strategy cluster_within_domain --num_train_epochs 0.001 --learning_rate 2e-05 --embedder nomic-ai/nomic-bert-2048 --clustering_model gtr_base --clustering_query_to_doc 1 --automatically_deduplicate_documents 0 --automatically_deduplicate_queries 0 --ddp_find_unused_parameters 1 --eval_rerank_topk 128 --lr_scheduler_type constant_with_warmup --warmup_steps 5600 --disable_dropout 1 --max_seq_length 64 --logging_steps 200 --train_cluster_size 64 --eval_cluster_size 256 --use_prefix 1 --transductive_corpus_size 16 --save_steps 99999999999 --logit_scale 50 --max_eval_batches 16 --torch_compile 0 --use_gc 1 --fp16 0 --bf16 1 --eval_steps 400000 --limit_layers 6 --max_batch_size_fits_in_memory 512 --disable_dropout 1 --arch biencoder --exp_name 2024-08-28-filter-sweep-stella-precompute --exp_group 2024-08-28-filter-sweep-precompute --use_wandb 0 --tiny_debug 1 --dataset nomic_supervised --hn_filter_model stella --dataset nomic_unsupervised --hn_tune_threshold 1 --hn_filter_precompute_vectors=1 --max_seq_length 32 --sampling_strategy cluster_within_domain --ddp_share_negatives_between_gpus 0 --max_batch_size_fits_in_memory 256