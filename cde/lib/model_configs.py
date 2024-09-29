# TODO: Save configs or push to hub. Make this nicer.
MODEL_FOLDER_DICT = {
   "cde--base--5epoch": "/fsx-checkpoints/jxm/cde/2024-08-06-transductive-pretrain-transductive-long-10node-3",
   "cde--filter--5epoch": "/fsx-checkpoints/jxm/cde/2024-08-10-transductive-pretrain-transductive-long-12node-filter",
   "cde--filter--1024--0.5epoch": "/fsx-checkpoints/jxm/cde/cde--unsupervised--1024--0.4",
   ######################
   "cde--supervised--nomic-0": "/fsx-checkpoints/jxm/cde/cde--supervised-0",
   "cde--supervised--nomic-1": "/fsx-checkpoints/jxm/cde/2024-09-18-supervised-final--epoch-5",
   "cde--small-v1": "/fsx-checkpoints/jxm/cde/2024-09-18-supervised-final-bge--epoch-4",
   ######################
   # Ran below supervised exp and found epoch 5 is highest MTEB, so using that going forward.
   "cde--supervised--nomic-1--epoch-1": "/fsx-checkpoints/jxm/cde/2024-09-18-supervised-final--epoch-1",
   "cde--supervised--nomic-1--epoch-2": "/fsx-checkpoints/jxm/cde/2024-09-18-supervised-final--epoch-2",
   "cde--supervised--nomic-1--epoch-3": "/fsx-checkpoints/jxm/cde/2024-09-18-supervised-final--epoch-3",
   "cde--supervised--nomic-1--epoch-4": "/fsx-checkpoints/jxm/cde/2024-09-18-supervised-final--epoch-4",
   "cde--supervised--nomic-1--epoch-5": "/fsx-checkpoints/jxm/cde/2024-09-18-supervised-final--epoch-5",
   #####################
   # Trying again with BGE data. (512 batch, 1 hard negative)
   "cde--supervised--nomic-2--epoch-1": "/fsx-checkpoints/jxm/cde/2024-09-18-supervised-final-bge--epoch-1",
   "cde--supervised--nomic-2--epoch-2": "/fsx-checkpoints/jxm/cde/2024-09-18-supervised-final-bge--epoch-2",
   "cde--supervised--nomic-2--epoch-3": "/fsx-checkpoints/jxm/cde/2024-09-18-supervised-final-bge--epoch-3",
   "cde--supervised--nomic-2--epoch-4": "/fsx-checkpoints/jxm/cde/2024-09-18-supervised-final-bge--epoch-4",
   "cde--supervised--nomic-2--epoch-5": "/fsx-checkpoints/jxm/cde/2024-09-18-supervised-final-bge--epoch-5",
   #####################
   # Slightly different setting for BGE. (4096 batch, no hard negatives)
   "cde--supervised--nomic-3--epoch-1": "/fsx-checkpoints/jxm/cde/2024-09-18-supervised-final-bge-big-batch-2--epoch-1",
   "cde--supervised--nomic-3--epoch-2": "/fsx-checkpoints/jxm/cde/2024-09-18-supervised-final-bge-big-batch-2--epoch-2",
   "cde--supervised--nomic-3--epoch-3": "/fsx-checkpoints/jxm/cde/2024-09-18-supervised-final-bge-big-batch-2--epoch-3",
   "cde--supervised--nomic-3--epoch-4": "/fsx-checkpoints/jxm/cde/2024-09-18-supervised-final-bge-big-batch-2--epoch-4",

   "cde--supervised--nomic-3--epoch-5": "/fsx-checkpoints/jxm/cde/2024-09-18-supervised-final-bge-big-batch-2--epoch-5",
   #####################
   # Hardest setting for BGE. (4096 batch, 1 hard negative)
   "cde--supervised--nomic-4--epoch-1": "/fsx-checkpoints/jxm/cde/2024-09-21-supervised-final-bge-big-batch-3--epoch-1",
   "cde--supervised--nomic-4--epoch-2": "/fsx-checkpoints/jxm/cde/2024-09-21-supervised-final-bge-big-batch-3--epoch-2",
   "cde--supervised--nomic-4--epoch-3": "/fsx-checkpoints/jxm/cde/2024-09-21-supervised-final-bge-big-batch-3--epoch-3",
   "cde--supervised--nomic-4--epoch-4": "/fsx-checkpoints/jxm/cde/2024-09-21-supervised-final-bge-big-batch-3--epoch-4",
   "cde--supervised--nomic-4--epoch-5": "/fsx-checkpoints/jxm/cde/2024-09-21-supervised-final-bge-big-batch-3--epoch-5",


   #####################
   # LLAMA models
   "cde--llama--save--test": "/fsx-checkpoints/jxm/cde/2024-09-24-supervised-final-bge-llama-7/",
   # this one was not bidirectional and padding was wrong
   "cde--supervised--llama-1--epoch-1": "/fsx-checkpoints/jxm/cde/2024-09-26-supervised-final-bge-llama-4-fsdp-test--epoch-1",
   # this one was bidirectional but had messed up input_ln
   "cde--supervised--llama-2--epoch-1": "/fsx-checkpoints/jxm/cde/2024-09-26-supervised-final-bge-llama-5-fsdp--epoch-1",
   "cde--supervised--llama-2--epoch-2": "/fsx-checkpoints/jxm/cde/2024-09-26-supervised-final-bge-llama-5-fsdp--epoch-2",
   # fixed input_ln
   "cde--supervised--llama-3--epoch-1": "/fsx-checkpoints/jxm/cde/2024-09-28-supervised-final-bge-llama-8-fsdp--epoch-1",
   "cde--supervised--llama-3--epoch-2": "/fsx-checkpoints/jxm/cde/2024-09-28-supervised-final-bge-llama-8-fsdp--epoch-2",
}