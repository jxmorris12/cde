from typing import Optional, List

import logging
import os
import pickle

import safetensors
import torch
import transformers
import wandb

from cde.collate import TokenizedCollator
from cde.lib import load_embedder_and_tokenizer, ContextualModelConfig
from cde.lib.model_configs import MODEL_FOLDER_DICT
from cde.model import get_model_class
# from cde.run_args import ModelArguments, DataArguments, TrainingArguments
from cde.trainer import CustomTrainer


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        module = module.replace("spider", "cde")
        return super().find_class(module, name)


class CustomPickle:
    Unpickler = CustomUnpickler



def load_trainer_from_checkpoint_and_args(
        model_folder: str, 
        beir_dataset_names: Optional[List[str]] = None,
        load_from_checkpoint: bool = True,
        return_args: bool = False,
        load_entire_trainer: bool = True,
    ):
    torch.compiler.reset()
    torch._dynamo.config.optimize_ddp = False
    torch._dynamo.config.cache_size_limit = 10_000

    # First, try reading from model folder.
    checkpoint_path = transformers.trainer_utils.get_last_checkpoint(
        model_folder
    )
    if os.path.exists(os.path.join(checkpoint_path, "data_args.bin")):
        data_args = torch.load(open(os.path.join(checkpoint_path, "data_args.bin"), "rb"), weights_only=False, pickle_module=CustomPickle)
        model_args = torch.load(open(os.path.join(checkpoint_path, "model_args.bin"), "rb"), weights_only=False, pickle_module=CustomPickle)
        training_args = torch.load(open(os.path.join(checkpoint_path, "training_args.bin"), "rb"), weights_only=False, pickle_module=CustomPickle)
        model_args.embedding_output_dim = None # backwards compatibility :-)
    else:
        raise RuntimeError("must have data_args.bin and model_args.bin available to load from disk")

    # Reset cached properties
    cached_keys = [k for k in training_args.__dict__.keys() if k.startswith("__cached")]
    for k in cached_keys:
        del training_args.__dict__[k]

    if not torch.cuda.is_available():
        from accelerate import PartialState
        print("[analyze_utils] No GPU available, loading model on CPU")
        training_args.use_cpu = True
        training_args._n_gpu = 0
        training_args.local_rank = -1  # Don't load in DDP
        training_args.distributed_state = PartialState(cpu=True)
        training_args.deepspeed_plugin = None  # For backwards compatibility
        training_args.bf16 = 0  # no bf16 in case no support from GPU

    transformers.set_seed(training_args.seed)
    logging.basicConfig(
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.WARNING
    )
    embedder_tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.embedder)
    dataset_backbone_tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.dataset_backbone or model_args.embedder,
            padding_side="right",
        )
    if not (dataset_backbone_tokenizer.pad_token) and dataset_backbone_tokenizer.bos_token:
        dataset_backbone_tokenizer.pad_token = dataset_backbone_tokenizer.bos_token
        print(f"Set pad token to bos token: {dataset_backbone_tokenizer.pad_token}")   
    dataset_backbone_tokenizer.add_eos_token = True
    beir_dataset_names = beir_dataset_names or []
    if training_args.tiny_debug: 
        beir_dataset_names = [ 'quora' ]
        training_args.max_eval_batches = 1

    retrieval_datasets = {}
    model_args.transductive_corpus_size = training_args.transductive_corpus_size
    # model_config = ContextualModelConfig(**vars(model_args))
    # model = model_cls(config=model_config)
    model_cls = get_model_class(model_args.architecture)
    model = model_cls.from_pretrained(checkpoint_path)
    model.eval()

    if not load_entire_trainer:
        if return_args:
            return model, (model_args, data_args, training_args)
        else:
            return model

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
        dataset_backbone_tokenizer=dataset_backbone_tokenizer,
        train_sampler_fn=None,
        eval_sampler_fns={},
        retrieval_datasets=retrieval_datasets,
    )
    # trainer._load_from_checkpoint(checkpoint_path)
    trainer.model = trainer.model.__class__.from_pretrained(checkpoint_path)
    trainer.model.eval()

    if return_args:
        return trainer, (model_args, data_args, training_args )
    else:
        return trainer
    

def load_model_from_alias(model_key: str) -> transformers.PreTrainedModel:
    model_folder = MODEL_FOLDER_DICT[model_key]
    print(f"Got key {model_key} - loading model from folder {model_folder}")
    trainer, (_, data_args, training_args) = load_trainer_from_checkpoint_and_args(
        model_folder=model_folder,
        load_from_checkpoint=True,
        return_args=True
    )
    trainer.model.eval()
    return trainer.model
