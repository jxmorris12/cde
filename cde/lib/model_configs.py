# TODO: Save configs or push to hub. Make this nicer.
MODEL_FOLDER_DICT = {
   "cde--base--5epoch": "/fsx-checkpoints/jxm/cde/2024-08-06-contextual-pretrain-contextual-long-10node-3",
   "cde--filter--5epoch": "/fsx-checkpoints/jxm/cde/2024-08-10-contextual-pretrain-contextual-long-12node-filter",
   "cde--filter--1024--0.5epoch": "/fsx-checkpoints/jxm/cde/cde--unsupervised--1024--0.4",
   ######################
   "cde--supervised--nomic-0": "/fsx-checkpoints/jxm/cde/cde--supervised-0",
   "cde--supervised--nomic-1": "/fsx-checkpoints/jxm/cde/2024-09-18-supervised-final--epoch-5",
   "cde--small-v1": "/fsx-checkpoints/jxm/cde/2024-09-18-supervised-final-bge--epoch-4",
   ######################
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
   # moved back to mean pooling, slightly better clustering, much larger batch size
   "cde--supervised--llama-4--epoch-1": "/fsx-checkpoints/jxm/cde/2024-09-29-supervised-final-bge-llama-11-fsdp--epoch-1/",
   # changed to 1b, changed prefixes
   "cde--supervised--llama1b-1--epoch-1": "/fsx-checkpoints/jxm/cde/2024-09-30-supervised-final-bge-llama-1b-2-fsdp--epoch-1/",
   "cde--supervised--llama1b-1--epoch-2": "/fsx-checkpoints/jxm/cde/2024-09-30-supervised-final-bge-llama-1b-2-fsdp--epoch-2/",
   # pretrained for .85 epoch w/ long prefixes, then sft for epochs
   "cde--supervised--llama1b-2--epoch-1": "/fsx-checkpoints/jxm/cde/2024-10-10-supervised-pretrain-llama-1b/",
   # pretrained for .85 epoch w/ long prefixes, then sft for epochs w/ better/fixed symmetry
   "cde--supervised--llama1b-3--epoch-1": "/fsx-checkpoints/jxm/cde/2024-10-10-supervised-pretrain-llama-1b-2--epoch-1/",
   # pretrained for .85 epoch w/ long prefixes without pooling over prefix
   "cde--supervised--llama1b-4--unsup": "/fsx-checkpoints/jxm/cde/2024-10-11-llama-1b-pretrain-2/",
   "cde--supervised--llama1b-4--epoch-1": "/fsx-checkpoints/jxm/cde/2024-10-14-supervised-pretrain-llama-1b-4--epoch-1/",
   "cde--supervised--llama1b-4--epoch-2": "/fsx-checkpoints/jxm/cde/2024-10-14-supervised-pretrain-llama-1b-4--epoch-2/",
   "cde--supervised--llama1b-4--epoch-3": "/fsx-checkpoints/jxm/cde/2024-10-14-supervised-pretrain-llama-1b-4--epoch-3/",
   "cde--supervised--llama1b-4--epoch-4": "/fsx-checkpoints/jxm/cde/2024-10-14-supervised-pretrain-llama-1b-4--epoch-4/",
   "cde--supervised--llama1b-4--epoch-5": "/fsx-checkpoints/jxm/cde/2024-10-14-supervised-pretrain-llama-1b-4--epoch-5/",
   # same as above but with more hard negatives
   "cde--supervised--llama1b-5--epoch-1": "/fsx-checkpoints/jxm/cde/2024-10-17-supervised-pretrain-llama-1b-5--epoch-1",
   "cde--supervised--llama1b-5--epoch-2": "/fsx-checkpoints/jxm/cde/2024-10-17-supervised-pretrain-llama-1b-5--epoch-2",
   # same as two above but with weighted-mean pooling
   "cde--supervised--llama1b-6--epoch-1": "/fsx-checkpoints/jxm/cde/2024-10-21-supervised-pretrain-llama-1b-5--epoch-1",
   "cde--supervised--llama1b-6--epoch-2": "/fsx-checkpoints/jxm/cde/2024-10-21-supervised-pretrain-llama-1b-5--epoch-2",
}