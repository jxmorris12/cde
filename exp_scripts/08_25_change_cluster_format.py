import submitit
import os
import sys


from cde.lib.change_clusters_format import convert_torch_to_txt

def run_job():
    # Create an executor using submitit's AutoExecutor
    executor = submitit.AutoExecutor(folder=os.path.join("log_submitit", "cluster"))
    # Set the number of CPUs per task
    num_cpus = 1  # Adjust as needed
    executor.update_parameters(
        cpus_per_task=num_cpus, 
        mem_gb=999,
        gpus_per_node=0, 
        timeout_min=24 * 60 * 5,
        slurm_account=os.environ["SLURM_ACCT_NAME"],
        slurm_partition=os.environ["SLURM_PARTITION_NAME"], 
        slurm_qos=os.environ["SLURM_QOS"]
    )
    # Prepare a list to hold the jobs
    jobs = []
    # Submit the function to be executed 16 times

    with executor.batch():
        for _ in range(5):
            # convert_torch_to_txt("data/cluster")
            job = executor.submit(convert_torch_to_txt, "data/cluster")
            jobs.append(job)
    
    # Wait for all jobs to complete and gather results
    print(f"Submitted jobs w/ ids {[job.job_id for job in jobs]}")

if __name__ == "__main__":
    run_job()
