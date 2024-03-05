# jxm 3/5/24
# automatically wraps stuff in torchrun

import os
import sys

import torch

TORCHRUN_BASE = "torchrun --nproc_per_node {num_gpus} finetune.py {command_args}"
def main():
    command_args = ' '.join(sys.argv[1:])
    torchrun_command = TORCHRUN_BASE.format(
        num_gpus=torch.cuda.device_count(),
        command_args=command_args
    )
    print(torchrun_command)
    os.system(torchrun_command)

if __name__ == '__main__':
    main()