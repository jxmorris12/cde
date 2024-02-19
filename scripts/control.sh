#!/bin/zsh
#SBATCH --job-name=tti3-reddit-method        # Job name
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks-per-node=1     # Number of tasks per node
#SBATCH --cpus-per-task=4       # Number of CPU cores per task
#SBATCH --mem=64G                # Memory per node
#SBATCH --time=240:00:00         # Time limit (hh:mm:ss)
#SBATCH --partition rush,nlplarge-sasha-highpri 
#SBATCH --gres gpu:1
#SBATCH --constraint "gpu-high"


python finetune.py --bf16=1 --use_wandb=1 --use_gc=0 --logging_steps=100 \
    --per_device_train_batch_size 64 --per_device_eval_batch_size 64 \
    --exp_name reddit-fake-2--64-token-noprefetch --use_wandb 1 \
    --num_train_epochs 100.0 --eval_steps 10000 --dataset_info "fake" \
    --torch_compile 1 --torch_compile_mode "max-autotune-no-cudagraphs"
