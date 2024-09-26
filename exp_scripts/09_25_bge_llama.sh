#!/bin/bash
#SBATCH -A memorization
#SBATCH -q memorization_high
#SBATCH --job-name=test_multinode
#SBATCH --output=log_submitit/multinode/llama/pretrain_%j.out
#SBATCH --error=log_submitit/multinode/llama/pretrain_%j.err
#SBATCH --nodes=16
#SBATCH --ntasks=16
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=7-00:00:00
#SBATCH --mem=900G
#SBATCH --requeue

# Training setup
GPUS_PER_NODE=8
# so processes know who to talk to
MAIN_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MAIN_PORT=6000

CMD=" \
    finetune.py --per_device_train_batch_size 64 --per_device_eval_batch_size 256 \
                --dataset nomic_supervised --sampling_strategy cluster_within_domain \
                 --num_train_epochs 5 --learning_rate 1e-4 \
                 --embedder nomic-ai/nomic-bert-2048 --clustering_model gtr_base \
                 --clustering_query_to_doc 1 \
                  --eval_rerank_topk 512 --lr_scheduler_type constant_with_warmup \
                 --warmup_steps 160 --max_seq_length 512 \
                 --logging_steps 1 --train_cluster_size 512 --eval_cluster_size 256 \
                 --use_prefix 1 --transductive_corpus_size 128 --transductive_tokens_per_document 4 \
                 --logit_scale 50 \
                 --max_eval_batches 16 --torch_compile 0 --use_gc 1 --fp16 0 --bf16 1 \
                 --eval_steps 5000 --disable_dropout 1 --arch transductive \
                 --exp_name 2024-09-26-supervised-final-bge-llama-4-fsdp-test \
                 --exp_group 2024-09-26-supervised-filter-filtered-llama-4-fsdp \
                 --hn_filter_model stella --hn_tune_threshold 1 \
                 --hn_filter_precompute_vector 0 \
                 --ddp_share_negatives_between_gpus 0 --num_hard_negatives 1 \
                 --num_eval_rerank_samples 1024 --save_strategy epoch --save_total_limit 5 \
                 --max_batch_size_fits_in_memory 2 \
                 --max_batch_size_fits_in_memory_first_stage 32 \
                 --use_wandb 1 --ddp_find_unused_parameters 1 --dataset bge \
                 --clustering_batch_packing_strategy random \
                 --dataset_backbone "unsloth/Meta-Llama-3.1-8B" \
                 --autoregressive_backbone 1 --transductive_sequence_dropout_prob 0.005
    "

# --save_steps 2 --save_strategy steps --max_seq_length 32 --transductive_corpus_size 8 --use_wandb 0 --dataset_backbone "gpt2"
# --save_steps 2 --save_strategy steps --use_wandb 1 --max_seq_length 32 --transductive_corpus_size 8 --use_wandb 0

LAUNCHER="accelerate launch \
    --config_file fsdp_config_16.yaml \
    --main_process_ip "$MAIN_ADDR" \
    --main_process_port $MAIN_PORT \
    --machine_rank \$SLURM_PROCID \
    --role $SLURMD_NODENAME: \
    --rdzv_conf rdzv_backend=c10d \
    --max_restarts 0 \
"

SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
"
srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "$LAUNCHER $CMD"

echo "END TIME: $(date)"
