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

# TODO: Finish this one day!

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
    --nproc_per_node 8 finetune.py \
    --per_device_train_batch_size 512 --per_device_eval_batch_size 256 --dataset nomic_supervised --sampling_strategy cluster_within_domain --num_train_epochs 3 --learning_rate 2e-05 --embedder nomic-ai/nomic-bert-2048 --clustering_model gtr_base --clustering_query_to_doc 1 --ddp_find_unused_parameters 0  --eval_rerank_topk 512 --transductive_tokens_per_document 5 --lr_scheduler_type linear --warmup_steps 20 --disable_dropout 1 --max_seq_length 512 --logging_steps 5 --train_cluster_size 512 --eval_cluster_size 256 --use_prefix 1 --transductive_corpus_size 512 --logit_scale 50 --max_eval_batches 16 --torch_compile 0 --use_gc 1 --fp16 0 --bf16 1 --eval_steps 5000 --disable_dropout 1 --arch transductive --exp_name 2024-09-21-supervised-final-bge-llama --exp_group 2024-09-21-supervised-filter-filtered-llama --hn_filter_model stella --hn_tune_threshold 1 --hn_filter_precompute_vector 0 --sampling_strategy cluster_within_domain --ddp_share_negatives_between_gpus 0 --num_hard_negatives 1 --num_eval_rerank_samples 1024 --save_strategy epoch --save_total_limit 5 --max_batch_size_fits_in_memory 4 --max_batch_size_fits_in_memory_first_stage 128 --use_wandb 0 --ddp_find_unused_parameters 1 --dataset bge --clustering_batch_packing_strategy tsp_greedy --use_wandb 1 --dataset_backbone "unsloth/Meta-Llama-3.1-8B" --autoregressive_backbone=1 --sampling_strategy cluster_within_domain --use_wandb 1 --hn_tune_threshold 1 --max_batch_size_fits_in_memory 1 --use_lora=0 --tiny_debug 0