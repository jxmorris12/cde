import logging
import os
import shlex

import datasets
import torch
import transformers
import wandb

from collate import TokenizerCollator
from dataset import (
    BeirDataset, NomicSupervisedDataset, NomicUnsupervisedDataset
)
from lib import load_embedder_and_tokenizer, ModelConfig
from model import get_model_class
from run_args import ModelArguments, DataArguments, TrainingArguments
from sampler import get_sampler
from trainer import CustomTrainer


args_str = "--per_device_train_batch_size 1024 --per_device_eval_batch_size 1024 --use_wandb 1 --bf16 1 --dataset nomic_unsupervised --sampling_strategy cluster_within_domain --exp_name 2024-04-04-transductive-28-dt-pos--gradclip --num_train_epochs 3 --learning_rate 2e-5 --embedder nomic-ai/nomic-bert-2048 --dataset_embedder nomic-ai/nomic-bert-2048 --clustering_model gtr_base --clustering_query_to_doc 1 --automatically_deduplicate_documents 1 --automatically_deduplicate_queries 1 --arch query_independent_dt --dataset_info batch --ddp_find_unused_parameters 0 --torch_compile 0 --eval_rerank_topk 1024 --lr_scheduler_type cosine --warmup_steps 20000 --disable_dropout 1 --eval_steps 10000 --max_seq_length 512 --max_batch_size_fits_in_memory 256 --use_gc 1 --logging_steps 40 --logit_scale 20.0"

model_folder = "/home/paperspace/tti3/saves/BACKUP--2024-04-04-transductive-28-dt-pos--gradclip/2024-04-04-transductive-28-dt-pos--gradclip/"




def main():
    # Helps with debugging.
    torch.compiler.reset()
    torch._dynamo.config.optimize_ddp = False
    torch._dynamo.config.cache_size_limit = 10_000

    os.environ["WANDB__SERVICE_WAIT"] = "30"

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = (
        parser.parse_args_into_dataclasses(
            shlex.split(args_str))
    )
    training_args.use_wandb = 0
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
        # https://github.com/beir-cellar/beir/blob/f062f038c4bfd19a8ca942a9910b1e0d218759d4/examples/dataset/download_dataset.py#L13
         'arguana',
         'webis-touche2020',
         'quora',
         'nfcorpus',
         'scidocs', 
         'scifact',
         'trec-covid',
         'signal1m',
         'fiqa',
         'trec-news',  
         'msmarco',
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
        beir_dataset_names = [ 'quora' ]
        training_args.max_eval_batches = 1

    beir_dict = {
        d: BeirDataset(dataset=d, embedder_rerank=model_args.embedder_rerank) 
        for d in sorted(beir_dataset_names)
    }
    retrieval_datasets = {
        **{f"BeIR/{k}": v for k,v in beir_dict.items()}
    }
    model_args.transductive_corpus_size = training_args.transductive_corpus_size
    model_config = ModelConfig(**vars(model_args))
    model_cls = get_model_class(model_args.architecture)
    model = model_cls(
        config=model_config,
        embedder=embedder,
        dataset_embedder=dataset_embedder,
        dataset_backbone=dataset_backbone,
    )

    collator = TokenizerCollator(
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
        train_dataset=None,
        eval_dataset=None,
        dataset_tokenizer=dataset_tokenizer,
        embedder_tokenizer=embedder_tokenizer,
        train_sampler=None,
        eval_samplers={},
        retrieval_datasets=retrieval_datasets,
    )
    checkpoint_path = transformers.trainer_utils.get_last_checkpoint(
        model_folder)
    trainer._load_from_checkpoint(checkpoint_path)
    trainer.evaluate_retrieval_datasets()

if __name__ == '__main__':
    main()
