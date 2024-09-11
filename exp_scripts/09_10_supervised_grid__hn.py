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
    timeout_min=(ONE_DAY_IN_MIN), 
    gpus_per_node=8, 
    cpus_per_task=64,
    slurm_array_parallelism=999,
    slurm_account=os.environ["SLURM_ACCT_NAME"],
    slurm_partition=os.environ["SLURM_PARTITION_NAME"], 
    slurm_qos=os.environ["SLURM_QOS"]
)

command_str = "torchrun --nproc_per_node 8 finetune.py --per_device_train_batch_size {batch_size} --per_device_eval_batch_size 256 --use_wandb 1 --dataset nomic_supervised --sampling_strategy cluster_within_domain --num_train_epochs 3 --learning_rate 2e-05 --embedder nomic-ai/nomic-bert-2048 --clustering_model gtr_base --clustering_query_to_doc 1 --ddp_find_unused_parameters 0 --eval_rerank_topk 128 --lr_scheduler_type constant_with_warmup --warmup_steps 5600 --disable_dropout 1 --max_seq_length 512 --logging_steps 200 --train_cluster_size {cluster_size} --eval_cluster_size 256 --use_prefix 1 --transductive_corpus_size 512 --save_strategy epoch --logit_scale 50 --max_eval_batches 16 --torch_compile 0 --use_gc 1 --fp16 0 --bf16 1 --eval_steps 400000 --max_batch_size_fits_in_memory 512 --disable_dropout 1 --arch transductive --exp_name 2024-09-10-filter-sweep-nomic-{cluster_size}-{batch_size}-{num_hard_negatives} --exp_group 2024-09-10-filter-sweep-nomic --use_wandb 1 --hn_filter_model nomic --hn_tune_threshold 1 --hn_filter_precompute_vectors=0 --sampling_strategy clustfer_within_domain --ddp_share_negatives_between_gpus 0 --max_batch_size_fits_in_memory 256 --use_wandb 1 --num_hard_negatives {num_hard_negatives}" 

args_dict = {
    "architecture": ["biencoder"],
    "sampling_strategy": ["cluster_within_domain"],
    "corpus_size": [512],
    ##############################################################
    "cluster_size": [64, 256, 1024, 4096, 16384, 131072, 131072*2, 131072*4, 131072*8, 131072*16, 131072*32],
    "batch_size": [4096, 2048, 1024, 512, 256],
    "num_hard_negatives": [1, 3, 7, 15],
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
        print(result_command)
        # job = executor.submit(function)
        # jobs.append(job)

print(f"*** SUBMITIT: Successfully submitted {len(jobs)} jobs. ***")

    