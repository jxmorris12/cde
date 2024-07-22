import itertools
import os
import shlex

import submitit

executor = submitit.AutoExecutor(folder="log_submitit")

# srun -A differential -q differential_high -p learn --gpus 8 --pty /bin/bash
# https://slurm.schedmd.com/squeue.html
# https://github.com/facebookincubator/submitit/blob/07f21fa1234e34151874c00d80c345e215af4967/submitit/slurm/slurm.py#L388

ONE_DAY_IN_MIN = 3 * 24 * 60
executor.update_parameters(
    timeout_min=(2 * ONE_DAY_IN_MIN), 
    gpus_per_node=8, 
    cpus_per_task=96,
    slurm_account=os.environ["SLURM_ACCT_NAME"],
    slurm_partition=os.environ["SLURM_PARTITION_NAME"], 
    slurm_qos=os.environ["SLURM_QOS"]
)

command_str = "torchrun --nproc_per_node 8 finetune.py --per_device_train_batch_size {batch_size} --per_device_eval_batch_size 256 --use_wandb 1 --dataset nomic_unsupervised --sampling_strategy cluster_within_domain --num_train_epochs 5 --learning_rate 1e-5 --embedder nomic-ai/nomic-bert-2048 --clustering_model gtr_base --clustering_query_to_doc 1 --automatically_deduplicate_documents 0 --automatically_deduplicate_queries 1 --arch {architecture} --ddp_find_unused_parameters 0 --eval_rerank_topk {corpus_size} --lr_scheduler_type linear --warmup_steps 5600 --disable_dropout 1 --max_seq_length 32 --logging_steps 2000 --train_cluster_size {cluster_size} --eval_cluster_size 256 --use_prefix 1 --transductive_corpus_size {corpus_size} --save_steps 16000 --logit_scale 50 --max_eval_batches 4 --exp_name 2024-07-15-biencoder-cluster-sweep--{batch_size}-{cluster_size} --exp_group 2024-07-15-biencoder-cluster-sweep --ddp_share_negatives_between_gpus 1 --torch_compile 0 --use_gc 1 --fp16 0 --bf16 1 --eval_steps 100000 --limit_layers 6 --sampling_strategy {sampling_strategy} --overwrite_output_dir --max_batch_size_fits_in_memory 1024" 

args_dict = {
    "architecture": ["biencoder"],
    "sampling_strategy": ["cluster_within_domain"],
    "corpus_size": [64],
    ##############################################################
    "batch_size": [64, 128, 512],
    "cluster_size": [16384, 4096, 1024, 256, 64],
}
combinations = list(itertools.product(*args_dict.values()))
args_list = [{key: value for key, value in zip(args_dict.keys(), combination)} for combination in combinations]

jobs = []
for idx, args in enumerate(args_list):
    print(f"Job {idx+1}/{len(args_list)}:", args)
    result_command = command_str.format(**args)
    function = submitit.helpers.CommandFunction(shlex.split(result_command))
    job = executor.submit(function)
    jobs.append(job)

print(f"*** SUBMITIT: Successfully submitted {len(jobs)} jobs. ***")
