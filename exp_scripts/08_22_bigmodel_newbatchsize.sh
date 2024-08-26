#!/bin/bash
#SBATCH -A differential
#SBATCH -q differential_high
#SBATCH --job-name=test_multinode
#SBATCH --output=log_submitit/multinode/pretrain_%j.out
#SBATCH --error=log_submitit/multinode/pretrain_%j.err
#SBATCH --nodes=12
#SBATCH --ntasks=12
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --time=5-00:00:00
#SBATCH --mem=192G
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
    --nnodes 12 \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint $head_node_ip:29500 \
    --nproc_per_node 8 finetune.py --per_device_train_batch_size 1024 --per_device_eval_batch_size 1024 --use_wandb 1 --dataset nomic_unsupervised --sampling_strategy cluster_within_domain --num_train_epochs 10 --learning_rate 1e-4 --embedder nomic-ai/nomic-bert-2048 --clustering_model gtr_base --clustering_query_to_doc 1 --automatically_deduplicate_documents 0 --automatically_deduplicate_queries 0 --arch transductive --ddp_find_unused_parameters 0 --eval_rerank_topk 512 --lr_scheduler_type linear --warmup_steps 5600 --disable_dropout 1 --max_seq_length 512 --train_cluster_size 256 --eval_cluster_size 256 --use_prefix 1 --transductive_corpus_size 512 --save_steps 10000 --save_strategy steps --logit_scale 50 --max_eval_batches 4 --exp_name 2024-08-22-transductive-pretrain-transductive-long-12node-filter --exp_group 2024-08-22--long-train-12node-filter --ddp_share_negatives_between_gpus 0 --torch_compile 0 --use_gc 1 --fp16 0 --bf16 1 --eval_steps 12000 --logging_steps 50  --use_wandb 1 --max_batch_size_fits_in_memory 128 --hn_tune_threshold 1.0 --ddp_find_unused_parameters 1
