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

from spider.collate import DocumentQueryCollatorWithPadding, TokenizerCollator
from spider.dataset import (
    load_synthetic_words_dataset, 
    BeirDataset, NomicSupervisedDataset, NomicUnsupervisedDataset
)
from spider.lib import get_rank, get_world_size, load_embedder_and_tokenizer, ModelConfig
from spider.model import get_model_class
from spider.run_args import ModelArguments, DataArguments, TrainingArguments
from spider.sampler import get_sampler
from spider.trainer import CustomTrainer


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
    torch.autograd.set_detect_anomaly(True)
    # torch.compile() settings.
    torch.compiler.reset()
    torch._dynamo.config.optimize_ddp = False
    # Following things are taken from 
    # https://github.com/pytorch/benchmark
    # to hopefully increase speed.
    # torch._dynamo.config.automatic_dynamic_shapes = False
    # torch._dynamo.config.force_parameter_static_shapes = False
    torch._dynamo.config.cache_size_limit = 10_000

    os.environ["WANDB__SERVICE_WAIT"] = "30"

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    transformers.set_seed(training_args.seed)
    datasets.logging.set_verbosity_error()
    logging.basicConfig(
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.WARNING
    )
    embedder, embedder_tokenizer = load_embedder_and_tokenizer(
        model_args.embedder,
    )

    beir_dataset_names = [
        # https://github.com/beir-cellar/beir/blob/f062f038c4bfd19a8ca942a9910b1e0d218759d4/examples/dataset/download_dataset.py#L13
        #  'arguana',
        #  'webis-touche2020',
        #  'quora',
        #  'nfcorpus',
        #  'scidocs', 
        #  'scifact',
        #  'trec-covid',
        #  'signal1m',
        #  'fiqa',
        #  'trec-news',  
        #  'msmarco',
    #############################################
         'nq',
        # 'cqadupstack',
    #############################################
        #  'bioasq', # huge (14m samples)
        #  'robust04',   
        #  'fever', # JSON parse error
        #  'dbpedia-entity',
        #  'hotpotqa',
        # 'climate-fever', # pyarrow.lib.ArrowIndexError: array slice would exceed array length
    ]
    if training_args.tiny_debug: 
        datasets.logging.set_verbosity_info()
        beir_dataset_names = [ 'quora' ]
        training_args.max_eval_batches = 1

    beir_dict = {
        d: BeirDataset(
            dataset=d, 
            embedder_rerank=model_args.embedder_rerank,
            use_prefix=data_args.use_prefix,
        ) 
        for d in sorted(beir_dataset_names)
    }
    retrieval_datasets = {
        **{f"BeIR/{k}": v for k,v in beir_dict.items()}
    }

    collator_cls = DocumentQueryCollatorWithPadding
    if data_args.dataset == 'synthetic_words':
        train_dataset, eval_dataset = load_synthetic_words_dataset()
    elif data_args.dataset == 'nomic_unsupervised':
        train_dataset = NomicUnsupervisedDataset(
            tokenizer=embedder_tokenizer,
            max_seq_length=model_args.max_seq_length,
            use_prefix=data_args.use_prefix,
        )
        eval_dataset = NomicSupervisedDataset(
            tokenizer=embedder_tokenizer,
            num_hard_negatives=0,
            max_seq_length=model_args.max_seq_length,
            use_prefix=data_args.use_prefix,
        )
        # eval_dataset = None
        # Need to tokenize and collate for this dataset
        collator_cls = TokenizerCollator
    elif data_args.dataset == 'nomic_supervised':
        train_dataset = NomicSupervisedDataset(
            tokenizer=embedder_tokenizer,
            num_hard_negatives=data_args.num_hard_negatives,
            max_seq_length=model_args.max_seq_length,
            use_prefix=data_args.use_prefix,
        )
        # Need to tokenize and collate for this dataset
        collator_cls = TokenizerCollator
    else:
        raise ValueError(f'Unsupported dataset {data_args.dataset}')
    

    if training_args.ddp_share_negatives_between_gpus:
        effective_train_batch_size = (training_args.per_device_train_batch_size * get_world_size())
    else:
        effective_train_batch_size = (training_args.per_device_train_batch_size)
    print(f"[*] loading sampler with effective_train_batch_size = {effective_train_batch_size}")
    train_sampler_fn = functools.partial(
        get_sampler,
        dataset=train_dataset,
        sampling_strategy=data_args.sampling_strategy,
        batch_size=effective_train_batch_size,
        cluster_size=data_args.train_cluster_size,
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
        cluster_size=data_args.eval_cluster_size,
        shuffle=False,
        clustering_model="gtr_base",
        clustering_query_to_doc=data_args.clustering_query_to_doc,
        num_samples=(training_args.per_device_eval_batch_size * training_args.max_eval_batches),
    )
    print("[main] creating val samplers")
    eval_sampler_fns = {
        "cluster_within_domain": functools.partial(eval_sampler_fn, sampling_strategy="cluster_within_domain"),
        "domain": functools.partial(eval_sampler_fn, sampling_strategy="domain"),
        "random": functools.partial(eval_sampler_fn, sampling_strategy="random"),
    }
    model_args.transductive_corpus_size = training_args.transductive_corpus_size
    model_config = ModelConfig(**vars(model_args))
    model_cls = get_model_class(model_args.architecture)
    if model_args.architecture == 'biencoder':
        model = model_cls(
            config=model_config,
            embedder=embedder,
        )
    else:
        dataset_backbone, dataset_tokenizer = load_embedder_and_tokenizer(
            model_args.embedder,
        )
        model = model_cls(
            config=model_config,
            embedder=embedder,
            dataset_backbone=dataset_backbone,
        )

    print("[main] creating collator")
    collator = collator_cls(
        tokenizer=embedder_tokenizer,
        padding='longest',
        return_tensors='pt',
        max_length=model_args.max_seq_length,
    )
    
    checkpoint = get_checkpoint(training_args)
    if get_rank() == 0:
        wandb_run_id = training_args.exp_name
        print("starting wandb run with name", wandb_run_id)
        wandb.init(
            entity="jack-morris",
            project="tti-nomic-5",
            name=wandb_run_id,
            resume=(checkpoint is not None),
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

    print("[main] creating trainer")
    trainer = CustomTrainer(
        data_collator=collator,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        # eval_dataset=eval_dataset,
        embedder_tokenizer=embedder_tokenizer,
        train_sampler_fn=train_sampler_fn,
        eval_sampler_fns={},
        # eval_sampler_fns=eval_sampler_fns,
        retrieval_datasets={},
        # retrieval_datasets=retrieval_datasets,
    )
    logging.info("train() loaded checkpoint %s", checkpoint)
    print("[main] trainer.train()")
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.evaluate_retrieval_datasets()


if __name__ == '__main__':
    main()
