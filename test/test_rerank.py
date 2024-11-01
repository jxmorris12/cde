import dataclasses
import shlex

import pytest
import torch
import transformers

from cde.collate import TokenizerCollator
from cde.dataset import BeirDataset
from cde.lib import ContextualModelConfig 
from cde.model import get_model_class
from cde.run_args import ModelArguments, DataArguments, TrainingArguments
from cde.trainer import CustomTrainer

from .helpers import FakeEmbedder

def load_fake_embedder_and_tokenizer():
    embedder = FakeEmbedder()
    tokenizer = transformers.AutoTokenizer.from_pretrained("sentence-transformers/gtr-t5-base")
    return embedder, tokenizer

@pytest.fixture
def fake_trainer():
    args_str = "--per_device_train_batch_size 8 --per_device_eval_batch_size 8 --use_wandb 0 --bf16 0 --dataset nomic_supervised --sampling_strategy cluster_within_domain --num_hard_negatives 7 --exp_name unsupervised-2-cluster-kmeans-test --num_train_epochs 1.2 --learning_rate 2e-6 --embedder \"sentence-transformers/gtr-t5-base\" --clustering_model gtr_base --clustering_query_to_doc 1 --architecture contextual --max_seq_length 32"
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = (
        parser.parse_args_into_dataclasses(
            shlex.split(args_str))
    )
    model_args.transductive_corpus_size = 20
    model_config = ContextualModelConfig(**vars(model_args))
    model_cls = get_model_class(model_args.architecture)
    embedder, tokenizer = load_fake_embedder_and_tokenizer()
    model = model_cls(
        config=model_config,
        embedder=embedder,
        dataset_backbone=embedder,
    )
    collator = TokenizerCollator(
        tokenizer=tokenizer,
        padding='longest',
        return_tensors='pt',
        max_length=model_args.max_seq_length,
    )
    beir_dataset_names = ["nfcorpus"]
    beir_dict = {
        d: BeirDataset(
            dataset=d, 
            embedder_rerank="sentence-transformers/gtr-t5-base",
            use_prefix=True,
        ) 
        for d in sorted(beir_dataset_names)
    }
    retrieval_datasets = {
        **{f"BeIR/{k}": v for k,v in beir_dict.items()}
    }
    trainer = CustomTrainer(
        data_collator=collator,
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=None,
        embedder_tokenizer=tokenizer,
        train_sampler_fn=None,
        eval_sampler_fns={},
        retrieval_datasets=retrieval_datasets,
    )
    trainer.model.eval()
    trainer.model = trainer.model.cpu()
    return trainer

def test_rerank_beir_default(fake_trainer):
    trainer = fake_trainer
    results_dict = trainer.evaluate_retrieval_datasets(n=16)
    print(results_dict)