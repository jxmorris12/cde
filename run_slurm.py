import itertools
from datetime import datetime

import torch
from slurmpy import Slurm


ACTUALLY_RUN_COMMAND = True

BASE_PYTHON_CMD = """
python finetune.py --per_device_train_batch_size 64 \
    --learning_rate 2e-5 \
    --lr_scheduler_type constant_with_warmup \
    --warmup_steps 25000 \
    --bf16=1 \
    --embedder sentence-transformers/gtr-t5-base \
    --eval_steps=25000 \
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
                "gres": "gpu:a6000:1",
                # "gres": "gpu:1",
                # "constraint": "a40|3090|a6000|a5000|a100-40",
                "ntasks": 1,
                "cpus-per-task": 4,
                "mem": "100G",
                "time": "168:00:00",  # 168 hours --> 1 week
                # "time": "504:00:00",  # 504 hours --> 3 weeks
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



now = datetime.now()
date_str = now.strftime("%Y-%m-%d")

for arch in ["query_dependent", "query_independent", "biencoder_extended", "biencoder"]:
    exp_name = date_str + "-" + arch
    cmd = BASE_PYTHON_CMD.format(ARCH=arch, EXP_NAME=exp_name)
    run_cmd(cmd=cmd, job_desc=exp_name)