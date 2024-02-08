from torch.utils.data import Dataset
import os
import logging
import os

import torch
import transformers
import wandb

from collate import DocumentQueryCollatorWithPadding
from dataset import load_reddit_train_and_val, load_synthetic_chars_dataset
from helpers import ModelConfig
from model import Model
from run_args import ModelArguments, DataArguments, TrainingArguments
from trainer import CustomTrainer

assert torch.cuda.device_count() > 0, "can't train without CUDA"


def main():
    # Helps with debugging.
    torch.autograd.set_detect_anomaly(True)
    torch.compiler.reset()
    # torch._logging.set_logs(dynamo=logging.DEBUG)
    # torch._dynamo.config.verbose = True

    # Allow W&B to start slowly.
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    os.environ["_WANDB_STARTUP_DEBUG"] = "true"

    # Fast data processing at risk of deadlocks...
    os.environ['TOKENIZERS_PARALLELISM'] = 'False'

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    transformers.set_seed(training_args.seed)
    logging.basicConfig(
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO
    )
    embedder = transformers.AutoModel.from_pretrained(model_args.embedder)
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
        'trec-covid',
        'signal1m',
        'robust04',
        # Other ones are certainly too big for repeated eval
        # 'webis-touche2020',
        # 'fever', 'quora',
    ]
    beir_dict = {} # TMP
    # beir_dict = {
    #     d: BeirDataset(dataset=d, embedder=model_args.embedder_rerank) for d in beir_dataset_names
    # }
    retrieval_datasets = {
        **{f"BeIR/{k}": v for k,v in beir_dict.items()}
    }
    for k,v in retrieval_datasets.items():
        v.tokenize(tokenizer=embedder_tokenizer, max_length=model_args.max_seq_length)


    if data_args.dataset == 'synthetic_chars':
        train_dataset, eval_dataset = load_synthetic_chars_dataset()
    elif data_args.dataset == 'reddit_supervised':
        train_dataset, eval_dataset = load_reddit_train_and_val(
            # data_folder="/home/jxm3/research/retrieval/tti3/data/mini",
            data_folder="/home/jxm3/research/retrieval/tti3/data/full",
            perc=0.95, 
            supervised=True,
        )
    elif data_args.dataset == 'reddit_unsupervised':
        train_dataset, eval_dataset = load_reddit_train_and_val(
            # data_folder="/home/jxm3/research/retrieval/tti3/data/mini",
            data_folder="/home/jxm3/research/retrieval/tti3/data/full",
            perc=0.95, 
            supervised=False,
        )
    else:
        raise ValueError(f'Unsupported dataset {data_args.dataset}')

    train_dataset.tokenize(tokenizer=embedder_tokenizer, max_length=model_args.max_seq_length)

    model_config = ModelConfig(**vars(model_args))
    dataset_embedder = transformers.AutoModel.from_pretrained(model_args.dataset_embedder)
    model = Model(config=model_config, embedder=embedder, dataset_embedder=dataset_embedder)
    wandb_run_id = training_args.exp_name
    print("starting wandb run with name", wandb_run_id)
    wandb.init(
        project="dataset-transformer-reddit",
        name=wandb_run_id,
        # resume=True,
    )
    wandb.watch(model)
    collator = DocumentQueryCollatorWithPadding(
        tokenizer=embedder_tokenizer,
        padding='longest',
        return_tensors='pt',
        max_length=model_args.max_seq_length,
    )
    trainer = CustomTrainer(
        data_collator=collator,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        embedder_tokenizer=embedder_tokenizer,
        retrieval_datasets=retrieval_datasets,
    )
    # trainer.evaluate_retrieval_datasets()
    trainer.train()


if __name__ == '__main__':
    main()
