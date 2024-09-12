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
    mem_gb=800,
    slurm_array_parallelism=8,
    slurm_account=os.environ["SLURM_ACCT_NAME"],
    slurm_partition=os.environ["SLURM_PARTITION_NAME"], 
    slurm_qos=os.environ["SLURM_QOS"]
)

command_str = "python evaluate_mteb.py cde--supervised--nomic-0" 

n_jobs = 16

jobs = []
with executor.batch():
    for idx in range(n_jobs):
        function = submitit.helpers.CommandFunction(shlex.split(command_str))
        job = executor.submit(function)
        jobs.append(job)

print(f"*** SUBMITIT: Successfully submitted {len(jobs)} jobs. ***")

    