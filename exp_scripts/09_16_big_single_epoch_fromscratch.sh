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
    --nproc_per_node 8 finetune.py --per_device_train_batch_size 1024 --per_device_eval_batch_size 256 --use_wandb 1 --dataset nomic_unsupervised --sampling_strategy cluster_within_domain --num_train_epochs 3 --learning_rate 2e-5 --embedder nomic-ai/nomic-bert-2048 --clustering_model gtr_base --clustering_query_to_doc 1 --arch transductive --ddp_find_unused_parameters 0 --eval_rerank_topk 1024 --lr_scheduler_type linear --warmup_steps 500 --disable_dropout 1 --max_seq_length 512 --train_cluster_size 1024 --eval_cluster_size 1024 --use_prefix 1 --transductive_corpus_size 512 --save_steps 400 --save_strategy steps --logit_scale 50 --max_eval_batches 4 --exp_name 2024-09-18-transductive-pretrain-transductive-long-12node-filter-fromscratch --exp_group 2024-09-16--long-train-12node-filter --ddp_share_negatives_between_gpus 0 --torch_compile 0 --use_gc 1 --fp16 0 --bf16 1 --eval_steps 1200000000000000000 --logging_steps 20  --use_wandb 1 --hn_tune_threshold 1.0 --ddp_find_unused_parameters 1 --transductive_corpus_size 1024 --transductive_tokens_per_document 4 --max_batch_size_fits_in_memory 16 --max_batch_size_fits_in_memory_first_stage 128 --hn_filter_model stella --seed 1777 --transductive_sequence_dropout_prob 0.005 --num_eval_rerank_samples 1024 --model_state_dict_from_path /opt/hpcaas/.mounts/fs-08829104cb559c481/jxm/cde/2024-09-16-transductive-pretrain-transductive-long-12node-filter-fromscratch
