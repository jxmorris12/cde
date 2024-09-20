import itertools
import os
import shlex

import submitit

executor = submitit.AutoExecutor(folder="log_submitit")

# srun -A differential -q differential_high -p learn --gpus 8 --pty /bin/bash
# https://slurm.schedmd.com/squeue.html
# https://github.com/facebookincubator/submitit/blob/07f21fa1234e34151874c00d80c345e215af4967/submitit/slurm/slurm.py#L388

THREE_DAYS_IN_MIN = 3 * 24 * 60
executor.update_parameters(
    timeout_min=THREE_DAYS_IN_MIN, 
    gpus_per_node=8, 
    cpus_per_task=64,
    mem_gb=800,
    slurm_array_parallelism=999,
    slurm_account=os.environ["SLURM_ACCT_NAME"],
    slurm_partition=os.environ["SLURM_PARTITION_NAME"], 
    slurm_qos=os.environ["SLURM_QOS"]
)

# This is just a super basic training command
#  that will train for a tiny amount of time so that clustering
#  runs.
command_str = "python finetune.py --per_device_train_batch_size {cluster_size} --per_device_eval_batch_size 256 --use_wandb 1 --dataset {dataset} --sampling_strategy cluster_within_domain --num_train_epochs 0.0000001 --learning_rate 1e-5 --embedder nomic-ai/nomic-bert-2048 --clustering_model {clustering_model} --clustering_query_to_doc 1 --automatically_deduplicate_documents 0 --automatically_deduplicate_queries 1 --arch transductive --ddp_find_unused_parameters 0 --eval_rerank_topk 64 --lr_scheduler_type linear --warmup_steps 5600 --disable_dropout 1 --max_seq_length 512 --logging_steps 2000 --train_cluster_size {cluster_size} --eval_cluster_size {cluster_size} --use_prefix 1 --transductive_corpus_size 256 --save_steps 50000 --logit_scale 50 --max_eval_batches 4 --exp_name 2024-07-07-transductive-pretrain-transductive--cluster --overwrite_output_dir --exp_group 2024-07-08--cluster --ddp_share_negatives_between_gpus 0 --torch_compile 0 --use_gc 1 --fp16 0 --bf16 1 --eval_steps 200000 --sampling_strategy cluster_within_domain --use_wandb 0 --max_batch_size_fits_in_memory 256 --tiny_debug 1 --save_steps 1000000 --max_seq_length 8 --use_wandb 0 --num_train_epochs 0.00001"  
# command_str = "python finetune.py --per_device_train_batch_size 256 --per_device_eval_batch_size 256 --use_wandb 1 --dataset nomic_unsupervised --sampling_strategy cluster_within_domain --num_train_epochs 0.0000001 --learning_rate 1e-5 --embedder nomic-ai/nomic-bert-2048 --clustering_model gtr_base --clustering_query_to_doc 1 --automatically_deduplicate_documents 0 --automatically_deduplicate_queries 1 --arch transductive --ddp_find_unused_parameters 0 --eval_rerank_topk 64 --lr_scheduler_type linear --warmup_steps 5600 --disable_dropout 1 --max_seq_length 512 --logging_steps 2000 --train_cluster_size {cluster_size} --eval_cluster_size {cluster_size} --use_prefix 1 --transductive_corpus_size 256 --save_steps 50000 --logit_scale 50 --max_eval_batches 4 --exp_name 2024-07-07-transductive-pretrain-transductive--cluster --overwrite_output_dir --exp_group 2024-07-08--cluster --ddp_share_negatives_between_gpus 0 --torch_compile 0 --use_gc 1 --fp16 0 --bf16 1 --eval_steps 200000 --sampling_strategy cluster_within_domain --use_wandb 0 --max_batch_size_fits_in_memory 256 --tiny_debug 1 --save_steps 1000000 --max_seq_length 8 --use_wandb 0 --num_train_epochs 0.00001"  

args_dict = {
    # "dataset": ["nomic_supervised", "nomic_unsupervised"],
    "dataset": ["nomic_unsupervised", "nomic_unsupervised", "bge"],
    "clustering_model": ["nomic", "bert", "sbert"],
    "cluster_size": [16384, 4096, 1024, 256, 64, 131072*8, 131072*16, 131072*32],
}
combinations = list(itertools.product(*args_dict.values()))
args_list = [{key: value for key, value in zip(args_dict.keys(), combination)} for combination in combinations]

jobs = []
with executor.batch():
    for idx, args in enumerate(args_list):
        print(f"Job {idx+1}/{len(args_list)}:", args)
        result_command = command_str.format(**args)
        function = submitit.helpers.CommandFunction(shlex.split(result_command))
        job = executor.submit(function)
        jobs.append(job)

print(f"*** SUBMITIT: Successfully submitted {len(jobs)} jobs. ***")
