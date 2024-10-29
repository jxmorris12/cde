from typing import Dict, List, Optional

import faiss # Need to import first to avoid errors :(

import copy
import functools
import os
import logging
import os

import datasets
import torch
import transformers
import wandb

from cde.collate import TokenizedCollator
from cde.dataset import (
    BeirDataset, 
    BGEDataset,
    NomicSupervisedDataset, 
    NomicUnsupervisedDataset
)
from cde.lib import (
    get_rank, 
    get_world_size, 
    ContextualModelConfig,
    print0
)
from cde.model import get_model_class
from cde.run_args import ModelArguments, DataArguments, TrainingArguments
from cde.sampler import get_sampler
from cde.trainer import CustomTrainer


try:
    torch.cuda.set_device(get_rank()) # Try to fix a DDP issue: https://github.com/pytorch/torchrec/issues/328
except RuntimeError: #No CUDA GPUs are available
    pass


logger = logging.getLogger(__name__)

def issue_warnings_after_load(load_result: Dict[str, List[str]]) -> None:
    if len(load_result.missing_keys) != 0:
        logger.warning(
            f"There were missing keys in the checkpoint model loaded: {load_result.missing_keys}."
        )
    if len(load_result.unexpected_keys) != 0:
        logger.warning(
            f"There were unexpected keys in the checkpoint model loaded: {load_result.unexpected_keys}."
        )


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
    # torch.compile() settings.
    torch.compiler.reset()
    # Following things are taken from 
    # https://github.com/pytorch/benchmark
    # to hopefully increase speed.
    # torch._dynamo.config.automatic_dynamic_shapes = False
    # torch._dynamo.config.force_parameter_static_shapes = False
    torch._dynamo.config.cache_size_limit = 32
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    torch._dynamo.config.optimize_ddp = False

    # https://github.com/pytorch/pytorch/issues/67978#issuecomment-1099316185
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    os.environ["WANDB__SERVICE_WAIT"] = "30"
    os.environ["TOKENIZERS_PARALLELISM"] = "0"

    # Higher DDP/FSDP log levels; optional.
    # os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    # os.environ["NCCL_DEBUG"] = "INFO"
    # os.environ["NCCL_DEBUG_SUBSYS"] = "INIT,COLL"


    # transformers.logging.set_verbosity_error()
    # datasets.logging.set_verbosity_error()
    datasets.utils.logging.enable_progress_bar()

    torch.set_float32_matmul_precision('high')

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.eval_strategy = "no"

    transformers.set_seed(training_args.seed)
    logging.basicConfig(
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.WARNING
    )
    embedder_tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.embedder,
        padding_side="right"
    )

    model_args.transductive_corpus_size = training_args.transductive_corpus_size
    model_config = ContextualModelConfig(**vars(model_args))
    model_cls = get_model_class(model_args.architecture)
    if model_args.architecture in ["biencoder", "dataset_prefix_biencoder"]:
        dataset_backbone_tokenizer = embedder_tokenizer
    else:
        dataset_backbone_tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.dataset_backbone or model_args.embedder,
            padding_side="right",
            add_eos_token=True,
            use_fast=True,
        )
    if not (dataset_backbone_tokenizer.pad_token) and dataset_backbone_tokenizer.bos_token:
        dataset_backbone_tokenizer.pad_token = dataset_backbone_tokenizer.bos_token
        print(f"Set pad token to bos token: {dataset_backbone_tokenizer.pad_token}")   

    model = model_cls(
        config=model_config,
    )

    if training_args.tiny_debug: 
        datasets.logging.set_verbosity_info()
        beir_dataset_names = [ 'arguana' ]
        training_args.max_eval_batches = 1
    else:
        beir_dataset_names = [
            # https://github.com/beir-cellar/beir/blob/f062f038c4bfd19a8ca942a9910b1e0d218759d4/examples/dataset/download_dataset.py#L13
            'arguana',
            # 'webis-touche2020',
            'quora',
            'nfcorpus',
            'scidocs', 
            'scifact',
            'trec-covid',
            # 'signal1m',
            'fiqa',
            # 'trec-news',  
            'msmarco',
        #############################################
            'nq',
        #############################################
        ]

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

    # if model_args.autoregressive_backbone:
    #     embed_eos = "</e>"
    #     if embed_eos not in dataset_backbone_tokenizer.vocab:
    #         dataset_backbone_tokenizer.add_tokens([embed_eos], special_tokens=True)
    #         model.second_stage_model.backbone.resize_token_embeddings(len(dataset_backbone_tokenizer))
        
    #     new_token_id = dataset_backbone_tokenizer.vocab[embed_eos]
    #     dataset_backbone_tokenizer.eos_token = embed_eos
    #     dataset_backbone_tokenizer.eos_token_id = new_token_id
    #     print0(f"[*] Added eos_token={embed_eos}, new len(tokenizer.vocab)={len(dataset_backbone_tokenizer)}")

    if data_args.dataset == 'nomic_unsupervised':
        train_dataset = NomicUnsupervisedDataset(
            tokenizer=dataset_backbone_tokenizer,
            first_stage_tokenizer=embedder_tokenizer,
            max_seq_length=model_args.max_seq_length,
            use_prefix=data_args.use_prefix,
            use_short_prefix=data_args.use_short_prefix,
            train_subdomain_key=data_args.train_subdomain_key,
        )
        eval_dataset = None
        # Need to tokenize and collate for this dataset
        collator_cls = TokenizedCollator
    elif data_args.dataset == 'nomic_supervised':
        train_dataset = NomicSupervisedDataset(
           tokenizer=dataset_backbone_tokenizer,
            first_stage_tokenizer=embedder_tokenizer,
            num_hard_negatives=data_args.num_hard_negatives,
            max_seq_length=model_args.max_seq_length,
            use_prefix=data_args.use_prefix,
        )
        eval_dataset = None
        collator_cls = TokenizedCollator
    elif data_args.dataset == 'bge':
        train_dataset = BGEDataset(
            tokenizer=dataset_backbone_tokenizer,
            first_stage_tokenizer=embedder_tokenizer,
            num_hard_negatives=data_args.num_hard_negatives,
            max_seq_length=model_args.max_seq_length,
            use_prefix=data_args.use_prefix,
            use_short_prefix=data_args.use_short_prefix,
        )
        eval_dataset = None
        collator_cls = TokenizedCollator
    else:
        raise ValueError(f'Unsupported dataset {data_args.dataset}')
    
    if training_args.ddp_share_negatives_between_gpus:
        effective_train_batch_size = (training_args.per_device_train_batch_size * get_world_size())
    else:
        effective_train_batch_size = (training_args.per_device_train_batch_size)
    print0(f"[*] loading sampler with effective_train_batch_size = {effective_train_batch_size}")
    train_sampler_fn = functools.partial(
        get_sampler,
        dataset=train_dataset,
        sampling_strategy=data_args.sampling_strategy,
        batch_size=training_args.per_device_train_batch_size,
        cluster_size=data_args.train_cluster_size,
        share_negatives_between_gpus=training_args.ddp_share_negatives_between_gpus,
        shuffle=True,
        clustering_model=data_args.clustering_model,
        downscale_and_normalize=data_args.clustering_downscale_and_normalize,
        batch_packing_strategy=data_args.clustering_batch_packing_strategy,
        clustering_query_to_doc=data_args.clustering_query_to_doc,
        seed=training_args.seed,
    )
    data_args_eval = copy.copy(data_args)
    data_args_eval.sampling_strategy = "domain" # always set this for eval
    eval_sampler_fn = functools.partial(
        get_sampler,
        dataset=(eval_dataset or train_dataset),
        batch_size=training_args.per_device_eval_batch_size,
        share_negatives_between_gpus=training_args.ddp_share_negatives_between_gpus,
        cluster_size=data_args.eval_cluster_size,
        shuffle=False,
        clustering_model="gtr_base",
        clustering_query_to_doc=data_args.clustering_query_to_doc,
        downscale_and_normalize=data_args.clustering_downscale_and_normalize,
        batch_packing_strategy=data_args.clustering_batch_packing_strategy,
        num_samples=(training_args.per_device_eval_batch_size * training_args.max_eval_batches),
    )
    print0("[main] creating val samplers")
    eval_sampler_fns = {
        "cluster_within_domain": functools.partial(eval_sampler_fn, sampling_strategy="cluster_within_domain"),
        "domain": functools.partial(eval_sampler_fn, sampling_strategy="domain"),
        "random": functools.partial(eval_sampler_fn, sampling_strategy="random"),
    }
    
    if training_args.model_state_dict_from_path:
        print0("[load_model] loading from path", training_args.model_state_dict_from_path)

        state_dict_checkpoint_folder = transformers.trainer_utils.get_last_checkpoint(training_args.model_state_dict_from_path)
        print0("[load_model] loading from folder", state_dict_checkpoint_folder)
        model = model.__class__.from_pretrained(
            state_dict_checkpoint_folder
        )

    print0("[main] creating collator")
    collator = collator_cls(
        tokenizer=embedder_tokenizer,
        padding='longest',
        return_tensors='pt',
        max_length=model_args.max_seq_length,
    )
    
    checkpoint = get_checkpoint(training_args)
    if get_rank() == 0:
        wandb_run_id = training_args.exp_name
        print0("starting wandb run with name", wandb_run_id)
        wandb.init(
            entity="jack-morris",
            project="tti-nomic-7",
            name=wandb_run_id,
            resume=False, # (checkpoint is not None),
            settings=wandb.Settings(symlink=False),
            mode=(None if training_args.use_wandb else "disabled")
        )
        wandb.config.update(
            {
                **vars(model_args),
                **vars(data_args),
                **vars(training_args),
            },
            allow_val_change=True,
        )
        # wandb.watch(model)
    

    print0("[main] creating trainer")
    if get_rank() == 0:
        # Print info stats for training on main worker
        transformers.logging.set_verbosity_info()


    trainer = CustomTrainer(
        data_collator=collator,
        model=model,
        args=training_args,
        data_args=data_args,
        model_args=model_args,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
        eval_dataset=None,
        dataset_backbone_tokenizer=dataset_backbone_tokenizer,
        train_sampler_fn=train_sampler_fn,
        eval_sampler_fns=eval_sampler_fns,
        retrieval_datasets=retrieval_datasets,
    )
    logging.info("train() loaded checkpoint %s", checkpoint)
    print0("[main] trainer.train()")

    # trainer.evaluate_retrieval_datasets()
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.evaluate_retrieval_datasets()


if __name__ == '__main__':
    main()
