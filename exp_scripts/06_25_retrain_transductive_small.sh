#!/bin/bash
#SBATCH --job-name=finetune
#SBATCH --partition=learnfair
#SBATCH --output=slurm_out/finetune_%j.out
#SBATCH --error=slurm_out/sbfinetune_%j.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=3-00:00
#SBATCH --mem=200G
#SBATCH --requeue

# module load python/3.8

torchrun --nproc_per_node 8 finetune.py \
    --per_device_train_batch_size 256 \                            
    --per_device_eval_batch_size 256 \
    --use_wandb 1 \
    --bf16 0 \
    --dataset nomic_unsupervised \
    --sampling_strategy cluster_within_domain \
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
    --transductive_corpus_size 64  \
    --save_steps 16000 \
    --logit_scale 50 \
    --max_eval_batches 4 \
    --exp_name 2024-06-25-transductive-pretrain-small \
    --ddp_share_negatives_between_gpus 0 \
    --use_wandb 0 --overwrite_output_dir --max_batch_size_fits_in_memory 128 --torch_compile 1 --use_gc 0 --fp16 1