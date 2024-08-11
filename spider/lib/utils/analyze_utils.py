from typing import Optional, List

import logging
import os

import torch
import transformers
import wandb

from spider.collate import TokenizedCollator
from spider.dataset import BeirDataset
from spider.lib import load_embedder_and_tokenizer, ModelConfig
from spider.model import get_model_class
from spider.run_args import ModelArguments, DataArguments, TrainingArguments
from spider.trainer import CustomTrainer


def load_trainer_from_checkpoint_and_args(
        model_folder: str, 
        beir_dataset_names: Optional[List[str]] = None,
        load_from_checkpoint: bool = True,
        return_args: bool = False
    ):
    torch.compiler.reset()
    torch._dynamo.config.optimize_ddp = False
    torch._dynamo.config.cache_size_limit = 10_000

    # First, try reading from model folder.
    checkpoint_path = transformers.trainer_utils.get_last_checkpoint(
        model_folder
    )
    if os.path.exists(os.path.join(checkpoint_path, "data_args.bin")):
        data_args = torch.load(open(os.path.join(checkpoint_path, "data_args.bin"), "rb"))
        model_args = torch.load(open(os.path.join(checkpoint_path, "model_args.bin"), "rb"))
        training_args = torch.load(open(os.path.join(checkpoint_path, "training_args.bin"), "rb"))
        model_args.embedding_output_dim = None # backwards compatibility :-)
        training_args.accelerator_config.gradient_accumulation_kwargs = None # backwards compatibility :-)
        training_args.use_wandb = 0
        training_args._n_gpu =  1 if torch.cuda.is_available() else 0  # Don't load in DDP
        training_args.local_rank = -1  # Don't load in DDP
        
        from accelerate.state import PartialState
        training_args.distributed_state = PartialState()
        # print("got device:", training_args.device, "//", training_args._setup_devices, training_args._n_gpu)
        # print("//", training_args.distributed_state.device)
        # TODO: Make this work proprly with DDP.
    else:
        raise RuntimeError("must have data_args.bin and model_args.bin available to load from disk")
    transformers.set_seed(training_args.seed)
    logging.basicConfig(
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.WARNING
    )
    embedder, embedder_tokenizer = load_embedder_and_tokenizer(
        model_args.embedder,
    )
    dataset_backbone, dataset_tokenizer = load_embedder_and_tokenizer(
        model_args.embedder,
    )

    beir_dataset_names = beir_dataset_names or []
    if training_args.tiny_debug: 
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

    collator = TokenizedCollator(
        tokenizer=embedder_tokenizer,
        padding='longest',
        return_tensors='pt',
        max_length=model_args.max_seq_length,
    )

    wandb.init(mode="disabled")
    trainer = CustomTrainer(
        data_collator=collator,
        model=model,
        args=training_args,
        data_args=data_args,
        model_args=model_args,
        train_dataset=None,
        eval_dataset=None,
        embedder_tokenizer=embedder_tokenizer,
        train_sampler_fn=None,
        eval_sampler_fns={},
        retrieval_datasets=retrieval_datasets,
    )
    if load_from_checkpoint:
        trainer._load_from_checkpoint(checkpoint_path)
    trainer.model.eval()

    if return_args:
        return trainer, (model_args, data_args, training_args )
    else:
        return trainer
    
