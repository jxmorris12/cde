from typing import Optional

import copy
import functools
import os
import logging
import os

import datasets
import torch
import transformers
import wandb

from collate import DocumentQueryCollatorWithPadding, TokenizerCollator
from dataset import (
    load_reddit_train_and_val, load_synthetic_words_dataset, 
    BeirDataset, NomicSupervisedDataset, NomicUnsupervisedDataset
)
from lib import get_rank, load_embedder_and_tokenizer, ModelConfig
from model import get_model_class
from run_args import ModelArguments, DataArguments, TrainingArguments
from sampler import get_sampler
from trainer import CustomTrainer


logger = logging.getLogger(__name__)

def get_checkpoint(training_args) -> Optional[str]:
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = transformers.trainer_utils.get_last_checkpoint(
            training_args.output_dir
        )
        if (
            last_checkpoint is None
            and len(os.listdir(training_args.output_dir)) > 0
        ):
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None
            and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    if checkpoint:
        logger.info("Loading from checkpoint %s", checkpoint)
    else:
        logger.info("No checkpoint found, training from scratch")

    return checkpoint


def main():
    # Helps with debugging.
    # torch.autograd.set_detect_anomaly(True)
    torch.compiler.reset()
    torch._dynamo.config.optimize_ddp = False
    # Following things are taken from 
    # https://github.com/pytorch/benchmark
    # to hopefully increase speed.
    # torch._dynamo.config.automatic_dynamic_shapes = False
    # torch._dynamo.config.force_parameter_static_shapes = False
    torch._dynamo.config.cache_size_limit = 10_000

    datasets.logging.set_verbosity_info()
    os.environ["WANDB__SERVICE_WAIT"] = "30"

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    transformers.set_seed(training_args.seed)
    logging.basicConfig(
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.WARNING
    )
    embedder, embedder_tokenizer = load_embedder_and_tokenizer(
        model_args.embedder,
    )
    dataset_embedder, dataset_tokenizer = load_embedder_and_tokenizer(
        model_args.dataset_embedder,
    )
    dataset_backbone, dataset_tokenizer = load_embedder_and_tokenizer(
        model_args.dataset_embedder,
    )

    beir_dataset_names = [
         'arguana',
         'webis-touche2020',
         'quora',
         'nfcorpus',
         'scidocs', 
         'scifact',
         'trec-covid',
         'signal1m',
         'fiqa',
    #############################################
        #  'robust04',   
        #  'trec-news',  
    #############################################
        #  'msmarco',
        #  'bioasq',
        # 'fever',
        # 'dbpedia',

    ]

    if training_args.tiny_debug: beir_dataset_names = [ 'quora' ]

    beir_dict = {
        d: BeirDataset(dataset=d, embedder_rerank=model_args.embedder_rerank) 
        for d in sorted(beir_dataset_names)
    }
    retrieval_datasets = {
        **{f"BeIR/{k}": v for k,v in beir_dict.items()}
    }

    collator_cls = DocumentQueryCollatorWithPadding
    if data_args.dataset == 'synthetic_words':
        train_dataset, eval_dataset = load_synthetic_words_dataset()
    elif data_args.dataset == 'reddit_supervised':
        train_dataset, eval_dataset = load_reddit_train_and_val(
            # data_folder="/home/jxm3/research/retrieval/tti3/data/mini",
            # data_folder="/home/jxm3/research/retrieval/tti3/data/full",
            data_folder="/home/jxm3/research/retrieval/tti3/data/full_t5",
            perc=0.98, 
            supervised=True,
        )
    elif data_args.dataset == 'reddit_unsupervised':
        train_dataset, eval_dataset = load_reddit_train_and_val(
            data_folder="/home/jxm3/research/retrieval/tti3/data/full",
            perc=0.98, 
            supervised=False,
        )
    elif data_args.dataset == 'nomic_unsupervised':
        train_dataset = NomicUnsupervisedDataset(
            tokenizer=embedder_tokenizer,
            max_seq_length=model_args.max_seq_length,
        )
        eval_dataset = NomicSupervisedDataset(
            tokenizer=embedder_tokenizer,
            num_hard_negatives=0,
            max_seq_length=model_args.max_seq_length,
        )
        # eval_dataset = None
        # Need to tokenize and collate for this dataset
        collator_cls = TokenizerCollator
    elif data_args.dataset == 'nomic':
        train_dataset = NomicSupervisedDataset(
            tokenizer=embedder_tokenizer,
            num_hard_negatives=data_args.num_hard_negatives,
            max_seq_length=model_args.max_seq_length,
        )
        # Need to tokenize and collate for this dataset
        collator_cls = TokenizerCollator
    else:
        raise ValueError(f'Unsupported dataset {data_args.dataset}')
    
    print("[*] loading sampler")
    train_sampler = get_sampler(
        dataset=train_dataset,
        sampling_strategy=data_args.sampling_strategy,
        batch_size=training_args.per_device_train_batch_size,
        cluster_size=224,
        shuffle=True,
        clustering_model=data_args.clustering_model,
        clustering_query_to_doc=data_args.clustering_query_to_doc,
    )
    data_args_eval = copy.copy(data_args)
    data_args_eval.sampling_strategy = "domain" # always set this for eval
    eval_sampler_fn = functools.partial(
        get_sampler,
        dataset=(eval_dataset or train_dataset),
        batch_size=training_args.per_device_eval_batch_size,
        cluster_size=224,
        shuffle=False,
        clustering_model="gtr_base",
        clustering_query_to_doc=data_args.clustering_query_to_doc,
        num_samples=(training_args.per_device_eval_batch_size * training_args.max_eval_batches),
    )
    eval_samplers = {
        "cluster_within_domain": eval_sampler_fn(sampling_strategy="cluster_within_domain"),
        "domain": eval_sampler_fn(sampling_strategy="domain"),
        "random": eval_sampler_fn(sampling_strategy="random"),
    }
    model_config = ModelConfig(**vars(model_args))
    model_cls = get_model_class(model_args.architecture)
    model = model_cls(
        config=model_config,
        embedder=embedder,
        dataset_embedder=dataset_embedder,
        dataset_backbone=dataset_backbone,
    )

    collator = collator_cls(
        tokenizer=embedder_tokenizer,
        padding='longest',
        return_tensors='pt',
        max_length=model_args.max_seq_length,
    )
    if get_rank() == 0:
        wandb_run_id = training_args.exp_name
        print("starting wandb run with name", wandb_run_id)
        wandb.init(
            entity="jack-morris",
            project="tti-nomic-4",
            name=wandb_run_id, # + "-RESUME",
    )
        wandb.config.update(
            {
                **vars(model_args),
                **vars(data_args),
                **vars(training_args),
            },
            allow_val_change=True,
        )
        wandb.watch(model)

    trainer = CustomTrainer(
        data_collator=collator,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_tokenizer=dataset_tokenizer,
        embedder_tokenizer=embedder_tokenizer,
        train_sampler=train_sampler,
        eval_samplers=eval_samplers,
        retrieval_datasets=retrieval_datasets,
    )
    checkpoint = get_checkpoint(training_args)
    logging.info("train() loaded checkpoint %s", checkpoint)
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.evaluate_retrieval_datasets(model=model)


if __name__ == '__main__':
    main()
