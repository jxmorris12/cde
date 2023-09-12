from torch.utils.data import Dataset
import pathlib, os, gzip, json
import logging
import os
import random

import torch
import transformers
import wandb

from dataset import BeirDataset, MsmarcoDatasetHardNegatives
from model import Model
from run_args import ModelArguments, DataTrainingArguments, TrainingArguments
from trainer import CustomTrainer


assert torch.cuda.device_count() > 0, "can't train without CUDA"

wandb.init(
    project="dataset-transformer",
    # name=exp_name,
    # id=kwargs_hash,
    # resume=True,
)

parser = transformers.HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()
transformers.set_seed(training_args.seed)

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

beir_dataset_names = [
    # these are the 5 smallest beir datasets...
    # 'arguana', # problem: query-doc IDs don't match? (TODO: investigate...)
    'nfcorpus',
    'scidocs', 
    'scifact',
    'fiqa',
    ########
    # 'trec-covid',
    # Other ones are certainly too big for repeated eval
    # 'webis-touche2020',
    # 'fever', 'quora',
]
beir_dict = {
    d: BeirDataset(dataset=d, embedder=model_args.embedder) for d in beir_dataset_names
}
retrieval_datasets = {
    **{f"BeIR/{k}": v for k,v in beir_dict.items()}
}

train_dataset = MsmarcoDatasetHardNegatives(
    embedder=model_args.embedder
)

embedder = transformers.AutoModel.from_pretrained(model_args.embedder)
embedder_tokenizer =  transformers.AutoTokenizer.from_pretrained(model_args.embedder)
for k,v in retrieval_datasets.items():
    v.tokenize(tokenizer=embedder_tokenizer, max_length=model_args.max_seq_length)

train_dataset.tokenize(tokenizer=embedder_tokenizer, max_length=model_args.max_seq_length)


# train_dataset = None
# trainer.evaluate_retrieval_datasets()

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

model = Model()
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=None,
    retrieval_datasets=retrieval_datasets,
)
trainer.train()

