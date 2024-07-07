#!/bin/bash
#SBATCH --job-name=finetune
#SBATCH --partition=learnfair
#SBATCH --output=slurm_out/pretrain_%j.out
#SBATCH --error=slurm_out/pretrain_%j.err
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --gres=gpu:8
#SBATCH --constraint volta32gb
#SBATCH --cpus-per-task=32
#SBATCH --time=3-00:00
#SBATCH --mem=120G
#SBATCH --requeue


# https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/slurm/sbatch_run.sh
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO
export NCCL_P2P_LEVEL=NVL
export NCCL_DEBUG=INFO
export HF_DATASETS_OFFLINE=1 
export HF_HUB_OFFLINE=1

srun torchrun \
    --nnodes 8 \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint $head_node_ip:29500 \
    --nproc_per_node 8 finetune.py --per_device_train_batch_size 256 \
    --per_device_eval_batch_size 256 \
    --use_wandb 1 \
    --bf16 1 --dataset nomic_unsupervised \
    --sampling_strategy random  \
    --num_train_epochs 1 \
    --learning_rate 1e-5 \
    --embedder distilbert-base-uncased \
    --clustering_model gtr_base \
    --clustering_query_to_doc 1 \
    --automatically_deduplicate_documents 0 \
    --automatically_deduplicate_queries 1 \
    --arch transductive \
    --ddp_find_unused_parameters 0 \
    --eval_rerank_topk 128 \
    --lr_scheduler_type linear \
    --warmup_steps 5600 \
    --disable_dropout 1 \
    --eval_steps 1200000000000 \
    --max_seq_length 64 \
    --use_gc 1 \
    --logging_steps 200 \
    --train_cluster_size 224 \
    --eval_cluster_size 224 \
    --use_prefix 1 \
    --transductive_corpus_size 32 \
    --save_steps 16000 \
    --logit_scale 50 \
    --max_eval_batches 4 \
    --exp_name 2024-06-26-transductive-pretrain-small--easy--4node-4 \
    --ddp_share_negatives_between_gpus 0 \
    --use_wandb 1 \
    --overwrite_output_dir \
    --max_batch_size_fits_in_memory 128 \
    --use_gc 0 --fp16 0
# --torch_compile 1 --torch_compile_mode max-autotune \
