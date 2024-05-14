import dataclasses
import shlex

import pytest
import torch
import transformers

from spider.collate import TokenizerCollator
from spider.dataset import BeirDataset
from spider.lib import ModelConfig 
from spider.model import get_model_class
from spider.run_args import ModelArguments, DataArguments, TrainingArguments
from spider.trainer import CustomTrainer


@dataclasses.dataclass
class FakeConfig:
    hidden_size = 32
    model_type = "fake"

class FakeModelOutput:
    def __init__(self, last_hidden_state: torch.Tensor):
        self.last_hidden_state = last_hidden_state

class FakeEmbedder(torch.nn.Module):
     def __init__(self, *args, **kwargs):
          import argparse
          self.config = FakeConfig()
          super().__init__()

     def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> FakeModelOutput:
        batch_size, seq_length = input_ids.shape
        last_hidden_state = torch.randn(
            batch_size, seq_length, self.config.hidden_size, device=input_ids.device)
        return FakeModelOutput(
            last_hidden_state=last_hidden_state
        )

def load_fake_embedder_and_tokenizer():
    embedder = FakeEmbedder()
    tokenizer = transformers.AutoTokenizer.from_pretrained("sentence-transformers/gtr-t5-base")
    return embedder, tokenizer

@pytest.fixture
def fake_trainer():
    args_str = "--per_device_train_batch_size 8 --per_device_eval_batch_size 8 --use_wandb 0 --bf16 0 --dataset nomic_supervised --sampling_strategy cluster_within_domain --num_hard_negatives 7 --exp_name unsupervised-2-cluster-kmeans-test --num_train_epochs 1.2 --learning_rate 2e-6 --embedder \"sentence-transformers/gtr-t5-base\" --clustering_model gtr_base --clustering_query_to_doc 1 --architecture transductive --max_seq_length 32"
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = (
        parser.parse_args_into_dataclasses(
            shlex.split(args_str))
    )
    model_config = ModelConfig(**vars(model_args))
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
    return trainer

def test_rerank_beir_default(fake_trainer):
    trainer = fake_trainer
    results_dict = trainer.evaluate_retrieval_datasets(n=16)
    print(results_dict)