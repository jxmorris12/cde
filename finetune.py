'''
This example shows how to train a SOTA Bi-Encoder for the MS Marco dataset (https://github.com/microsoft/MSMARCO-Passage-Ranking).
The model is trained using hard negatives which were specially mined with different dense and lexical search methods for MSMARCO. 

This example has been taken from here with few modifications to train SBERT (MSMARCO-v3) models: 
(https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/ms_marco/train_bi-encoder-v3.py)

The queries and passages are passed independently to the transformer network to produce fixed sized embeddings.
These embeddings can then be compared using cosine-similarity to find matching passages for a given query.

For training, we use MultipleNegativesRankingLoss. There, we pass triplets in the format:
(query, positive_passage, negative_passage)

Negative passage are hard negative examples, that were mined using different dense embedding methods and lexical search methods.
Each positive and negative passage comes with a score from a Cross-Encoder. This allows denoising, i.e. removing false negative
passages that are actually relevant for the query.

Running this script:
python train_msmarco_v3.py
'''

from torch.utils.data import Dataset
import pathlib, os, gzip, json
import logging
import random

import transformers

from dataset import MSMarcoDataset
from model import Model
from run_args import ModelArguments, DataTrainingArguments, TrainingArguments
from trainer import CustomTrainer


parser = transformers.HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)


train_dataset = MSMarcoDataset(embedder="sentence-transformers/gtr-t5-base")

model = Model()
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=None,
)
trainer.train()

