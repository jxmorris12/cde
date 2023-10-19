import itertools
from datetime import datetime

import torch
from slurmpy import Slurm


ACTUALLY_RUN_COMMAND = True


# Total batch size for contrastive loss:
#   >> per_device_train_batch_size * (1 + num_hard_negatives)
BASE_PYTHON_CMD = """
python finetune.py \
    --logging_steps 100 \
    --per_device_train_batch_size 32 \
    --num_hard_negatives 32 \
    --use_gc 1 \
    --max_batch_size_fits_in_memory 512 \
    --learning_rate 2e-5 \
    --lr_scheduler_type constant_with_warmup \
    --num_train_epochs 20 \
    --warmup_steps 20000 \
    --bf16=1 \
    --use_wandb 1 \
    --embedder "sentence-transformers/msmarco-distilbert-base-tas-b" \
    --backbone "distilbert-base-uncased" \
    --eval_steps=5000 \
    --evaluation_strategy steps \
    --architecture {ARCH} \
    --exp_name {EXP_NAME}
"""


def run_cmd(cmd: str, job_desc: str):
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    job_name = f"{dt_string} {job_desc}"
    ##
    print("job_name >>", job_name)
    print("cmd >>", cmd.strip())
    ##

    if ACTUALLY_RUN_COMMAND:
        slurm = Slurm(
            job_name,
            slurm_kwargs={
                "partition": "rush",
                # "partition": "gpu",
                "gres": "gpu:a6000:1",
                # "constraint": "a40|3090|a6000|a5000|a100-40",
                "ntasks": 1,
                "cpus-per-task": 4,
                "mem": "64G",
                "time": "336:00:00",  # 336 hours --> 2 weeks
            },
            slurm_flags=[
                "requeue",
            ],
        )
        slurm.run(
            f"""
        {cmd}
        """
        )
    ##
    print("\n\n")


# NAME_STR = ""
NAME_STR = "more-hard-negatives-16"

now = datetime.now()
date_str = now.strftime("%Y-%m-%d")

for arch in ["query_dependent", "query_independent", "biencoder_extended", "biencoder"]:
    exp_name = NAME_STR + arch
    cmd = BASE_PYTHON_CMD.format(ARCH=arch, EXP_NAME=exp_name)
    run_cmd(cmd=cmd, job_desc=exp_name)
