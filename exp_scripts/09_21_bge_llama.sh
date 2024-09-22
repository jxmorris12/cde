#!/bin/bash
#SBATCH -A memorization
#SBATCH -q memorization_high
#SBATCH --job-name=test_multinode
#SBATCH --output=log_submitit/multinode_llama/pretrain_%j.out
#SBATCH --error=log_submitit/multinode_llama/pretrain_%j.err
#SBATCH --nodes=3
#SBATCH --ntasks=3
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --time=7-00:00:00
#SBATCH --mem=800G
#SBATCH --requeue

# Training setup
GPUS_PER_NODE=8
# so processes know who to talk to
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000
NNODES=$SLURM_NNODES
NODE_RANK=$SLURM_PROCID 
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CMD=" \
    finetune.py --per_device_train_batch_size 512 --per_device_eval_batch_size 256 \
                --dataset nomic_supervised --sampling_strategy cluster_within_domain \
                 --num_train_epochs 5 --learning_rate 2e-05 \
                 --embedder nomic-ai/nomic-bert-2048 --clustering_model gtr_base \
                 --clustering_query_to_doc 1 --ddp_find_unused_parameters 0 \
                  --eval_rerank_topk 512 --lr_scheduler_type constant_with_warmup \
                --warmup_steps 200 --disable_dropout 1 --max_seq_length 512 \
                 --logging_steps 50 --train_cluster_size 512 --eval_cluster_size 256 \
                 --use_prefix 1 --transductive_corpus_size 512 --logit_scale 50 \
                 --max_eval_batches 16 --torch_compile 0 --use_gc 1 --fp16 0 --bf16 1 \
                 --eval_steps 5000 --disable_dropout 1 --arch transductive \
                 --exp_name 2024-09-21-supervised-final-bge-llama \
                 --exp_group 2024-09-21-supervised-filter-filtered-llama \
                 --hn_filter_model stella --hn_tune_threshold 1 \
                 --hn_filter_precompute_vector 0 \
                 --sampling_strategy cluster_within_domain \
                 --ddp_share_negatives_between_gpus 0 --num_hard_negatives 1 \
                 --num_eval_rerank_samples 1024 --save_strategy epoch --save_total_limit 5 \
                 --max_batch_size_fits_in_memory 128 \
                 --max_batch_size_fits_in_memory_first_stage 4 \
                 --use_wandb 1 --ddp_find_unused_parameters 1 --dataset bge \
                 --clustering_batch_packing_strategy tsp_greedy \
                 --dataset_backbone "meta-llama/Meta-Llama-3-8B" --use_wandb 0 \
                --autoregressive_backbone=1
    "

LAUNCHER="accelerate launch \
    --multi_gpu \
    --num_machines $NNODES \
    --num_processes $WORLD_SIZE \
    --main_process_ip "$MASTER_ADDR" \
    --main_process_port $MASTER_PORT \
    --num_processes $WORLD_SIZE \
    --machine_rank \$SLURM_PROCID \
    --role $SLURMD_NODENAME: \
    --rdzv_conf rdzv_backend=c10d \
    --max_restarts 0 \
    --tee 3 \
"

SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "

clear; srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "$LAUNCHER $CMD" 2>&1 | tee ~/logs/%x-%j.txt

echo "END TIME: $(date)"