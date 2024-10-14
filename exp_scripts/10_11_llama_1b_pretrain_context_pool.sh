#!/bin/bash
#SBATCH -A differential
#SBATCH -q differential_high
#SBATCH --job-name=test_multinode
#SBATCH --output=log_submitit/multinode/llama/pretrain_%j.out
#SBATCH --error=log_submitit/multinode/llama/pretrain_%j.err
#SBATCH --nodes=12
#SBATCH --ntasks=12
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --time=7-00:00:00
#SBATCH --mem=900G
#SBATCH --requeue

export HF_DATASETS_OFFLINE=1 

# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=INIT,COLL


# Training setup
GPUS_PER_NODE=8
# so processes know who to talk to
MAIN_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MAIN_PORT=6000

CMD="finetune.py --per_device_train_batch_size 512 --per_device_eval_batch_size 256 \
                 --dataset nomic_unsupervised --sampling_strategy cluster_within_domain \
                 --num_train_epochs 3 --learning_rate 2e-5 \
                 --embedder nomic-ai/nomic-bert-2048 --clustering_model mixedbread \
                 --clustering_query_to_doc 1 \
                  --eval_rerank_topk 512 --lr_scheduler_type constant_with_warmup \
                 --warmup_steps 400 --max_seq_length 512 \
                 --logging_steps 1 --train_cluster_size 512 --eval_cluster_size 512 \
                 --use_prefix 1 --transductive_corpus_size 512 --transductive_tokens_per_document 2 \
                 --logit_scale 50 \
                 --max_eval_batches 16 --torch_compile 0 --use_gc 1 --fp16 0 --bf16 1 \
                 --eval_steps 999999999 --disable_dropout 0 --arch transductive \
                 --exp_name 2024-10-11-llama-1b-pretrain-3-pool-context \
                 --exp_group 2024-10-11-llama-1b-pretrain-3 \
                 --hn_filter_model stella --hn_tune_threshold 1 \
                 --hn_filter_precompute_vector 0 \
                 --ddp_share_negatives_between_gpus 0 --num_hard_negatives 0 \
                 --num_eval_rerank_samples 1024 --save_strategy steps --save_steps 400 --save_total_limit 2 \
                 --max_batch_size_fits_in_memory 16 \
                 --max_batch_size_fits_in_memory_first_stage 64 \
                 --use_wandb 1 --ddp_find_unused_parameters 1 \
                 --dataset_backbone meta-llama/Llama-3.2-1B \
                 --autoregressive_backbone 1 --transductive_sequence_dropout_prob 0.005 \
                 --pooling_strategy mean --use_short_prefix 0 \
                 --pool_ignore_instruction_tokens 1 --pool_ignore_contextual_tokens 0
    "

#  --clustering_batch_packing_strategy tsp_greedy \
# --save_steps 2 --save_strategy steps --use_wandb 1 --max_seq_length 32 --transductive_corpus_size 8 --use_wandb 0

LAUNCHER="accelerate launch \
    --config_file fsdp_config_12.yaml \
    --main_process_ip "$MAIN_ADDR" \
    --main_process_port $MAIN_PORT \
    --machine_rank \$SLURM_PROCID \
    --role $SLURMD_NODENAME: \
    --rdzv_conf rdzv_backend=c10d \
    --max_restarts 0 \
"

SRUN_ARGS=" \
    --wait=600 \
    --kill-on-bad-exit=1 \
"
srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "$LAUNCHER $CMD"

echo "END TIME: $(date)"
