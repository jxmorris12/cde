import itertools
import os
import shlex

import submitit

executor = submitit.AutoExecutor(folder="log_submitit")

# srun -A differential -q differential_high -p learn --gpus 8 --pty /bin/bash
# https://slurm.schedmd.com/squeue.html
# https://github.com/facebookincubator/submitit/blob/07f21fa1234e34151874c00d80c345e215af4967/submitit/slurm/slurm.py#L388

ONE_DAY_IN_MIN = 24 * 60
executor.update_parameters(
    timeout_min=(3 * ONE_DAY_IN_MIN), 
    gpus_per_node=8, 
    cpus_per_task=96,
    slurm_array_parallelism=10,
    slurm_account=os.environ["SLURM_ACCT_NAME"],
    slurm_partition=os.environ["SLURM_PARTITION_NAME"], 
    slurm_qos=os.environ["SLURM_QOS"]
)

command_str = "torchrun --nproc_per_node 8 finetune.py --per_device_train_batch_size 1024 --per_device_eval_batch_size 256 --use_wandb 1 --dataset nomic_unsupervised --sampling_strategy cluster_within_domain --num_train_epochs 3 --learning_rate 2e-05 --embedder nomic-ai/nomic-bert-2048 --clustering_model gtr_base --clustering_query_to_doc 1 --automatically_deduplicate_documents 0 --automatically_deduplicate_queries 0 --ddp_find_unused_parameters 1 --eval_rerank_topk 128 --lr_scheduler_type constant_with_warmup --warmup_steps 5600 --disable_dropout 1 --max_seq_length 64 --logging_steps 200 --train_cluster_size 256 --eval_cluster_size 256 --use_prefix 1 --transductive_corpus_size 16 --save_steps 99999999999 --logit_scale 50 --max_eval_batches 16 --torch_compile 0 --use_gc 1 --fp16 0 --bf16 1 --eval_steps 400000 --limit_layers 6 --max_batch_size_fits_in_memory 512 --disable_dropout 1 --arch biencoder --exp_name 2024-08-28-filter-sweep-{hn_filter_model} --exp_group 2024-08-28-filter-sweep --use_wandb 0 --hn_filter_model {hn_filter_model} --dataset nomic_unsupervised --hn_tune_threshold 1 --hn_filter_precompute_vectors=1 --max_seq_length 32 --sampling_strategy cluster_within_domain --ddp_share_negatives_between_gpus 0 --max_batch_size_fits_in_memory 256" 
# command_str = "torchrun --nproc_per_node 8 finetune.py --per_device_train_batch_size 1024 --per_device_eval_batch_size 256 --use_wandb 1 --dataset nomic_unsupervised --num_train_epochs 3 --learning_rate 2e-05 --embedder nomic-ai/nomic-bert-2048 --clustering_model gtr_base --clustering_query_to_doc 1 --automatically_deduplicate_documents 0 --automatically_deduplicate_queries 0 --ddp_find_unused_parameters 1 --eval_rerank_topk 128 --lr_scheduler_type constant_with_warmup --warmup_steps 5600 --disable_dropout 1 --max_seq_length 64 --logging_steps 200 --train_cluster_size 256 --eval_cluster_size 256 --use_prefix 1 --transductive_corpus_size 16 --save_steps 99999999999 --logit_scale 50 --max_eval_batches 16 --torch_compile 0 --use_gc 1 --fp16 0 --bf16 1 --eval_steps 400000 --limit_layers 6 --max_batch_size_fits_in_memory 512 --disable_dropout 1 --arch biencoder --exp_name 2024-08-28-filter-sweep-sbert --exp_group 2024-08-28-filter-sweep --hn_filter_model sbert --dataset nomic_unsupervised --hn_tune_threshold 1 --hn_filter_precompute_vectors=1 --max_seq_length 32 --sampling_strategy cluster_within_domain --ddp_share_negatives_between_gpus 0 --max_batch_size_fits_in_memory 256" 

args_dict = {
    ##############################################################
    "hn_filter_model": ["nomic", "sbert", "sentence_t5"],
    ##############################################################
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

print(f"*** SUBMITIT: Successfully submitted {len(jobs)} jobs. ***") #  {[job.id for id in jobs <-- TODO: How to add?

    