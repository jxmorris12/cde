#!/bin/bash
#SBATCH -A memorization
#SBATCH -q memorization_high
#SBATCH --job-name=test_multinode
#SBATCH --output=log_submitit/multinode/pretrain_%j.out
#SBATCH --error=log_submitit/multinode/pretrain_%j.err
#SBATCH --nodes=16
#SBATCH --ntasks=16
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --time=7-00:00:00
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
    --nnodes 16 \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint $head_node_ip:29500 \
    --nproc_per_node 8 finetune.py --per_device_train_batch_size 512 --per_device_eval_batch_size 256 --use_wandb 1 --dataset nomic_unsupervised --sampling_strategy cluster_within_domain --num_train_epochs 5 --learning_rate 1e-4 --embedder nomic-ai/nomic-bert-2048 --clustering_model gtr_base --clustering_query_to_doc 1 --arch biencoder --ddp_find_unused_parameters 0 --eval_rerank_topk 512 --lr_scheduler_type linear --warmup_steps 5000 --disable_dropout 1 --max_seq_length 512 --train_cluster_size 256 --eval_cluster_size 256 --use_prefix 1 --save_steps 500 --save_strategy steps --logit_scale 50 --max_eval_batches 4 --exp_name 2024-09-20-biencoder-pretrain-5epoch --exp_group 2024-09-20-biencoder-pretrain-5epoch --ddp_share_negatives_between_gpus 0 --torch_compile 0 --use_gc 1 --fp16 0 --bf16 1 --eval_steps 1200000000 --logging_steps 50 --use_wandb 1 --hn_tune_threshold 1.0 --ddp_find_unused_parameters 1 --max_batch_size_fits_in_memory 256 --hn_filter_model nomic --save_strategy epoch --save_total_limit 5