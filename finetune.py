from torch.utils.data import Dataset
import pathlib, os, gzip, json
import logging
import os
import random

import torch
import transformers
import wandb

from collate import DocumentQueryCollatorWithPadding
from dataset import BeirDataset, MsmarcoDatasetHardNegatives
from helpers import ModelConfig
from model import Model
from run_args import ModelArguments, DataArguments, TrainingArguments
from trainer import CustomTrainer


assert torch.cuda.device_count() > 0, "can't train without CUDA"

parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()
transformers.set_seed(training_args.seed)

wandb_run_id = training_args.exp_name
print("starting wandb run with name", wandb_run_id)
wandb.init(
    project="dataset-transformer",
    name=wandb_run_id,
    # resume=True,
)

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
embedder = transformers.AutoModel.from_pretrained(model_args.embedder).encoder
embedder_tokenizer =  transformers.AutoTokenizer.from_pretrained(model_args.embedder)

beir_dataset_names = [
    # these are the 5 smallest beir datasets...
    # 'arguana', # problem: query-doc IDs don't match? (TODO: investigate...)
    'nfcorpus',
    'scidocs', 
    'scifact',
    'fiqa',
    ########
    'msmarco', # this is the *real* eval set...
    # 'trec-covid',
    # Other ones are certainly too big for repeated eval
    # 'webis-touche2020',
    # 'fever', 'quora',
]
# beir_dict = {
#     d: BeirDataset(dataset=d, embedder=model_args.embedder) for d in beir_dataset_names
# }
# retrieval_datasets = {
#     **{f"BeIR/{k}": v for k,v in beir_dict.items()}
# }
# for k,v in retrieval_datasets.items():
#     v.tokenize(tokenizer=embedder_tokenizer, max_length=model_args.max_seq_length)

train_dataset = MsmarcoDatasetHardNegatives(
    embedder=model_args.embedder,
    num_hard_negatives=data_args.num_hard_negatives,
)
train_dataset.tokenize(tokenizer=embedder_tokenizer, max_length=model_args.max_seq_length)


# train_dataset = None
# trainer.evaluate_retrieval_datasets()

# Allow W&B to start slowly.
os.environ["WANDB__SERVICE_WAIT"] = "300"
os.environ["_WANDB_STARTUP_DEBUG"] = "true"

# Prevent deadlocks...
# os.environ['TOKENIZERS_PARALLELISM'] = 'false'

model_config = ModelConfig(**vars(model_args))
backbone = transformers.AutoModel.from_pretrained(model_args.backbone)
model = Model(config=model_config, embedder=embedder, backbone=backbone)
collator = DocumentQueryCollatorWithPadding(
    tokenizer=embedder_tokenizer,
    padding='longest',
    return_tensors='pt'
)
trainer = CustomTrainer(
    data_collator=collator,
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=None,
    # retrieval_datasets=retrieval_datasets,
    retrieval_datasets={},
)
trainer.train()

