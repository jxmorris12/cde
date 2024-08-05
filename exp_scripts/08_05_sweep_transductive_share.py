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
    slurm_array_parallelism=12,
    slurm_account=os.environ["SLURM_ACCT_NAME"],
    slurm_partition=os.environ["SLURM_PARTITION_NAME"], 
    slurm_qos=os.environ["SLURM_QOS"]
)

command_str = "torchrun --nproc_per_node 8 finetune.py --per_device_train_batch_size {batch_size} --per_device_eval_batch_size 256 --use_wandb 1 --dataset nomic_unsupervised --sampling_strategy cluster_within_domain --num_train_epochs {epochs} --learning_rate {learning_rate} --embedder nomic-ai/nomic-bert-2048 --clustering_model gtr_base --clustering_query_to_doc 1 --automatically_deduplicate_documents 0 --automatically_deduplicate_queries 0 --arch {architecture} --ddp_find_unused_parameters 0 --eval_rerank_topk 512 --lr_scheduler_type constant_with_warmup --warmup_steps 5600 --disable_dropout 1 --max_seq_length 32 --logging_steps 2000 --train_cluster_size {cluster_size} --eval_cluster_size 256 --use_prefix 1 --transductive_corpus_size {corpus_size} --save_steps 99999999999 --logit_scale 50 --max_eval_batches 16 --exp_name 2024-08-05-transductive-share-cluster-sweep--{batch_size}-{cluster_size} --exp_group 2024-08-03-transductive-share-cluster-sweep --ddp_share_negatives_between_gpus 1 --torch_compile 0 --use_gc 1 --fp16 0 --bf16 1 --eval_steps 100000 --limit_layers 6 --sampling_strategy {sampling_strategy} --overwrite_output_dir --max_batch_size_fits_in_memory 2048" 

args_dict = {
    "architecture": ["transductive"],
    "sampling_strategy": ["cluster_within_domain"],
    "corpus_size": [32],
    ##############################################################
    "cluster_size": [64, 256, 1024, 4096, 16384, 131072, 131072*2, 131072*4], #, 131072*8, 131072*16, 131072*32],
    # "cluster_size": [64],
    "batch_size": [512, 256, 128, 64, 32],
    # "batch_size": [256],
    ##############################################################
    "epochs": [3],
    "learning_rate": [2e-5],
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
