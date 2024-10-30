from typing import Any, Dict, List, Optional, Union, Tuple

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
from cde.dataset import FineWeb, FineWebEdu
from cde.lib import (
    get_rank, get_world_size, 
    load_embedder_and_tokenizer, 
    load_model_state_dict_from_path, 
    ContextualModelConfig,
    print0
)
from cde.model import get_model_class
from cde.run_args import ModelArguments, DataArguments, TrainingArguments
from cde.sampler import get_sampler


# 
# sample command:
# > torchrun --nproc_per_node 8 scripts/run_mlm_finewebedu.py \
#     --exp_name fineweb-test --exp_group fineweb-test-group \
#     --learning_rate 5e-4 --adam_beta2 0.98 \
#     --per_device_train_batch_size 128 --gradient_accumulation_steps 8 \
#     --ddp_find_unused_parameters 0 --logging_steps 40 --use_wandb 1 \
#     --disable_dropout 0 --bf16 1 --transductive_corpus_size 512 \
#     --lr_scheduler_type linear
# 

class MlmTrainer(transformers.Trainer):

    def __init__(self, args, data_args, model_args, tokenizer, train_sampler_fn, **kwargs):
        super().__init__(**kwargs)
        self.args = args
        self.data_args = data_args
        self.model_args = model_args
        self.tokenizer = tokenizer
        self._train_sampler_fn = train_sampler_fn

        # track classifier head in here since we'll get rid of it anyway.
        if hasattr(self.model, "second_stage_model"):
            try:
                config = self.model.second_stage_model.backbone.config
            except AttributeError:
                config = self.model.second_stage_model.config
        else:
            config = self.model.embedder.config

        self.classifier = torch.nn.Linear(
            config.n_embd, config.vocab_size
        )
        self.classifier.to(args.device)

        self.mlm_probability = args.mlm_probability
    
    def _get_train_sampler(self) -> torch.utils.data.Sampler:
        return self._train_sampler_fn()
    
    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) 
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long, device=self.args.device)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
        
    def compute_loss(
        self,
        model: transformers.PreTrainedModel,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
    ) -> Union[Tuple[torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor]:

        # Rename keys
        input_ids = inputs.pop("text_input_ids")
        unmasked_input_ids = input_ids.clone()

        input_ids, labels = self.torch_mask_tokens(input_ids)

        inputs["input_ids"] = input_ids
        inputs["attention_mask"] = inputs.pop("text_attention_mask")

        # Sample fewer inputs if batch size is too large
        effective_batch_size = len(inputs["input_ids"])
        full_transductive_corpus_size = self.model.config.transductive_corpus_size * self.model.config.transductive_tokens_per_document

        # Randomly reorder dataset input ids.
        R1 = torch.randperm(effective_batch_size)[:full_transductive_corpus_size]

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            dataset_input_ids=unmasked_input_ids,
            # dataset_input_ids=unmasked_input_ids[R1],
            # dataset_input_ids=input_ids["input_ids"][R1],
            dataset_attention_mask=inputs["attention_mask"],
            # dataset_attention_mask=inputs["attention_mask"][R1],
            output_hidden_states=True,
        )
        logits = self.classifier(outputs["hidden_states"])

        mlm_loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), 
            labels.reshape(-1), 
            ignore_index=-100
        )
        # print(f"mlm_loss: {mlm_loss}")

        return mlm_loss

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

    # Higher DDP log level.
    # os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

    # transformers.logging.set_verbosity_error()
    # datasets.logging.set_verbosity_error()
    datasets.utils.logging.disable_progress_bar()

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
    embedder, embedder_tokenizer = load_embedder_and_tokenizer(
        model_args.embedder,
    )

    if data_args.dataset.startswith("finewebedu"):
        train_dataset = FineWebEdu(
            tokenizer=embedder_tokenizer,
            max_seq_length=model_args.max_seq_length,
            tiny=data_args.dataset.endswith("-tiny"),
        )
    elif data_args.dataset.startswith("fineweb"):
        train_dataset = FineWeb(
            tokenizer=embedder_tokenizer,
            max_seq_length=model_args.max_seq_length,
            tiny=data_args.dataset.endswith("-tiny"),
        )
    else:
        raise ValueError(f"Invalid dataset for MLM pretraining {args.dataset}")


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
        clustering_query_to_doc=data_args.clustering_query_to_doc,
        seed=training_args.seed,
    )
    model_args.transductive_corpus_size = training_args.transductive_corpus_size
    model_config = ContextualModelConfig(**vars(model_args))
    model_cls = get_model_class(model_args.architecture)
    if model_args.architecture in ['biencoder', 'dataset_prefix_biencoder', 'contextual_cross_attention']:
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
    
    if training_args.model_state_dict_from_path:
        print0("[load_model] loading from path", training_args.model_state_dict_from_path)
        state_dict = load_model_state_dict_from_path(
            training_args.model_state_dict_from_path
        )
        # del state_dict["second_stage_model._orig_mod.output_projection.2.weight"]
        # del state_dict["second_stage_model._orig_mod.output_projection.2.bias"]
        load_result = model.load_state_dict(state_dict, strict=False)
        issue_warnings_after_load(load_result)

    print0("[run_mlm_fineweb] creating collator")
    collator = TokenizedCollator(
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
            project="contextual-pretrain-0",
            name=wandb_run_id,
            resume=False, # (checkpoint is not None),
        )
        wandb.config.update(
            {
                **vars(model_args),
                **vars(data_args),
                **vars(training_args),
            },
            allow_val_change=True,
        )
    

    print0("[run_mlm_fineweb] creating trainer")
    if get_rank() == 0:
        # Print info stats for training on main worker
        transformers.logging.set_verbosity_info()
    
    trainer = MlmTrainer(
        data_collator=collator,
        model=model,
        args=training_args,
        data_args=data_args,
        model_args=model_args,
        train_dataset=train_dataset,
        tokenizer=embedder_tokenizer,
        train_sampler_fn=train_sampler_fn,
    )
    logging.info("train() loaded checkpoint %s", checkpoint)
    print0("[run_mlm_fineweb] trainer.train()")
    trainer.train(resume_from_checkpoint=checkpoint)


if __name__ == '__main__':
    main()
