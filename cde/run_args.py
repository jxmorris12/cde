from typing import Optional

from dataclasses import dataclass, field

import datetime
import logging
import math
import os

import torch
import transformers


from cde.lib import (
    count_cpus,
    exit_if_running_or_finished_wandb,
    get_rank, 
    get_world_size, 
    print0,
)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    disable_dropout: bool = field(
        default=False,
        metadata={"help": "set model dropout rate to zero"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    max_seq_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length for tokenizer"}
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    embedder: str = field(
        default="nomic-ai/nomic-embed-text-v1-unsupervised",
        metadata={"help": "embedder name for the model (encoder)"}
    )
    dataset_backbone: Optional[str] = field(
        default=None,
        metadata={"help": "embedder name for the model (output embedding model)"}
    )
    embedder_rerank: str = field(
        default="sentence-transformers/gtr-t5-base",
        metadata={"help": "embedder name for reranking"}
    )
    architecture: str = field(
        default="biencoder",
        metadata = {
            "choices": ["biencoder", "biencoder_plus_plus", "dataset_prefix_biencoder", "transductive", "transductive__encoder_decoder", "two_head_mlp", "contextual_cross_attention"],
        }
    )
    pooling_strategy: str = field(
        default="mean",
        metadata = {
            "choices": ["mean", "mean_weighted", "last_token"],
        }
    )
    limit_layers_first_stage: Optional[int] = field(
        default=None,
        metadata={
            "help": "If set, will load backbone and embedders with limited number of layers (first stage model)"
        }
    )
    limit_layers: Optional[int] = field(
        default=None,
        metadata={
            "help": "If set, will load backbone and embedders with limited number of layers"
        }
    )
    autoregressive_backbone: bool = field(
        default=False,
        metadata={
            "help": "Whether to use autoregressive backbone"
        }
    )
    logit_scale: float = field(
        default=50,
        metadata={
            "help": "temperature for contrastive learning",
        }
    )
    disable_transductive_rotary_embedding: bool = field(
        default=True,
        metadata={
            "help": "Whether to disable rotary embedding on the transductive part of that model"
        }
    )
    embedding_output_dim: Optional[int] = field(
        default=None,
        metadata={
            "help": "If set, sets bottleneck dim for contrastive outputs"
        }
    )
    transductive_sequence_dropout_prob: float = field(
        default=0.0,
        metadata={
            "help": "Sequence dropout val for transductive model"
        }
    )
    transductive_tokens_per_document: int = field(
        default=1,
        metadata={
            "help": "Number of tokens per document for transductive model"
        }
    )
    transductive_tie_token_embeddings: bool = field(
        default=False,
        metadata={
            "help": "Whether to tie token embeddings for transductive model"
        }
    )
    pool_ignore_instruction_tokens: bool = field(
        default=False,
        metadata={
            "help": "Whether to ignore instruction tokens when pooling. Only applies in the long-format case."
        }
    )
    pool_ignore_contextual_tokens: bool = field(
        default=False,
        metadata={
            "help": "Whether to ignore contextual tokens when pooling."
        }
    )
    def __post_init__(self):
        if self.transductive_tokens_per_document > 1:
            assert self.architecture in ["transductive", "contextual_cross_attention"], "transductive_tokens_per_document only works with transductive architectures"


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset: str = field(
        default="nomic_supervised", 
        metadata={
            "help": "The name of the dataset to use:",
            "choices": ["synthetic_words", "nomic_supervised", "nomic_unsupervised", "fineweb-tiny", "finewebedu-tiny", "fineweb", "finewebedu", "bge"]
        }
    )
    sampling_strategy: str = field(
        default="random",
        metadata={
            "help": "sampling strategy for batches",
            "choices": ["random", "domain", "cluster", "cluster_within_domain"]
        }
    )
    clustering_model: str = field(
        default="gtr_base",
        metadata={
            "help": "Model to use for clustering",
            "choices": ["bm25", "gtr_base", "bert", "sbert", "mixedbread"],
        }
    )
    clustering_query_to_doc: bool = field(
        default=True,
        metadata={
            "help": "Whether to use query-doc weightings for clustering, or just queries",
        }
    )
    clustering_downscale_and_normalize: bool = field(
        default=True,
        metadata={
            "help": "Whether to downscale and then normalize embeddings before clustering",
        }
    )
    clustering_batch_packing_strategy: str = field(
        default="random",
        metadata = {
            "choices": ["random", "tsp_greedy"],
        }
    )
    num_hard_negatives: int = field(
        default=0,
        metadata={
            "help": "Number of hard negatives for training",
        }
    )
    train_cluster_size: int = field(
        default=256,
        metadata={
            "help": "Cluster size for train data",
        }
    )
    train_subdomain_key: Optional[str] = field(
        default=None,
        metadata={
            "help": "Subdomain to filter within train data (only if set)",
        }
    )
    eval_cluster_size: int = field(
        default=256,
        metadata={
            "help": "Cluster size for val data",
        }
    )
    use_prefix: bool = field(
        default=True,
        metadata={
            "help": "Whether to add domain-specific prefixes to data"
        }
    )
    use_short_prefix: bool = field(
        default=True,
        metadata={
            "help": "Whether to use short prefixes for the BGE dataset, or the long ones, when prefixes are enabled"
        }
    )
    def __post_init__(self):
        pass


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    transductive_input_strategy: str = field(
        default="topk",
        metadata={
            "help": "dataset-level info to feed into dataset-conditioned model",
            "choices": ["fake", "dummy", "random_corpus", "topk", "random_corpus__topk__interp", "topk_pool", "null_topk", "null"],
        }
    )
    transductive_n_outputs_ensemble: int = field(
        default=1,
        metadata={
            "help": "Number of heads(?) to use, if doing ensembling over top-k outputs"
        }
    )
    # https://github.com/huggingface/transformers/blob/e82c1cb78e178519060b9391214727be75a218ca/src/transformers/training_args.py#L121
    output_dir: str = field(
        default="saves",
        metadata={"help": "Output directory for training saves"}
    )
    use_gc: bool = field(
        default=False,
        metadata={"help": "whether to use GradCache"}
    )
    num_train_epochs: float = field(
        default=3.0, 
        metadata={
            "required": False,
            "help": "Number of epochs for training"
        },
    )
    hn_tune_threshold: Optional[float] = field(
        default=None,
        metadata={
            "help": "threshold for tuning hard negatives (in 0-1) or none",
        }
    )
    hn_filter_model: str = field(
        default="nomic",
        metadata={
            "help": "model to use for filtering hard negatives",
            "choices": ["nomic", "stella", "sbert", "sentence_t5"],
        }
    )
    hn_filter_precompute_vectors: bool = field(
        default=False,
        metadata={
            "help": "whether to precompute embeddings for hard negative filtering",
        }
    )
    learning_rate: float = field(
        default=2e-5,
        metadata={"help": "The initial learning rate for AdamW on the backbone model."}
    )
    remove_unused_columns: Optional[bool] = field(
        default=False, metadata={"help": "Remove columns not required by the model when using an nlp.Dataset."}
    )
    per_device_train_batch_size: int = field(
        default=64, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=64, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    transductive_corpus_size: int = field(
        default=256,
        metadata={
            "help": "Corpus input size for transductive encoder",
        }
    )
    evaluation_strategy: str = "steps"
    logging_steps: int = field(
        default=200, 
        metadata={"help": "Log every X updates steps."}
    )
    eval_steps: int = field(
        default=30_000, 
        metadata={"help": "Run an evaluation every X steps."}
    )
    max_batch_size_fits_in_memory: int = field(
        default=64, 
        metadata={"help": "Max batch size for contrastive learning in grad cache"}
    )
    max_batch_size_fits_in_memory_first_stage: Optional[int] = field(
        default=None, 
        metadata={"help": "Max batch size for contrastive learning in grad cache (first-stage onnly)"}
    )
    eval_rerank_topk: int = field(
        default=128,
         metadata={"help": "Number of reranked examples during eval"}
    )
    save_strategy: str = field(
        default="steps",
    )
    save_steps: int = field(
        default=4000,
    )
    save_total_limit: int = field(
        default=1,
        metadata={"help": "Max number of checkpoints to save"}
    )

    exp_name: str = field(
        default=None,
        metadata={
            "help": "Name for this experiment, unique string",
            "required": "True",
        }
    )
    exp_group: str = field(
        default="",
        metadata={
            "help": "Group name for multiple experiments, a string",
            "required": "False",
        }
    )

    lr_scheduler_type: str = field(
        default="constant_with_warmup",
        metadata={
            "help": "Type of scheduler to use"
        }
    )
    warmup_steps: int = field(
        default=8_000,
        metadata={
            "help": "Linear warmup over warmup_steps."
        }
    )
    weight_decay: float = field(
        default=0.01, 
        metadata={"help": "The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights in [`AdamW`] optimizer."}
    )
    automatically_deduplicate_documents: bool = field(
        default=False,
        metadata={"help": "whether to count duplicate queries as joint positive samples"}
    ) # TODO: Remove (no longer used).
    automatically_deduplicate_queries: bool = field(
        default=False,
        metadata={"help": "whether to count duplicate queries as joint positive samples"}
    ) # TODO: Remove (no longer used).
    ddp_find_unused_parameters: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "When using distributed training, the value of the flag `find_unused_parameters` passed to "
                "`DistributedDataParallel`."
            )
        },
    )
    ddp_share_negatives_between_gpus: bool = field(
        default=False,
        metadata={
            "help": "Whether to share negative examples between GPUs (alters batch size)"
        }
    )
    max_eval_batches: int = field(
        default=16,
        metadata={
            "help": "Max batches to use for eval (to reduce noise)"
        }
    )
    tiny_debug: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Run in jack's tiny debug mode. Disables eval."
            )
        },
    )
    model_state_dict_from_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "If set, will load model from weights within a checkpoint in this folder"
        }
    )
    # apparently this default is slightly more efficient:
    # https://github.com/pytorch/pytorch/issues/118421
    ddp_bucket_cap_mb: Optional[int] = field(
        default=100,
        metadata={
            "help": (
                "When using distributed training, the value of the flag `bucket_cap_mb` passed to "
                "`DistributedDataParallel`."
            )
        },
    )
    mlm_probability: float = field(
        default=0.30, 
        metadata={"help": "Ratio of tokens to mask for MLM training."}
    )
    wandb_exit_if_running_or_finished: bool = field(
        default=False,
        metadata={"help": "Whether to only run if previous run crashed"}
    )
    use_wandb: bool = field(
        default=False, metadata={"help": "Whether or not to log to Weights & Biases."}
    )
    report_to: str = "wandb"
    num_eval_rerank_samples: int = field(
        default=64,
        metadata={"help": "Number of samples to rerank during eval"}
    )

    def __setattr__(self, name, value):
        super(transformers.TrainingArguments, self).__setattr__(name, value)

    def __post_init__(self):
        super().__post_init__()
        ############################################################################
        if self.use_wandb:
            self.report_to = ["wandb"] if (self.local_rank <= 0) else []
        else:
            self.report_to = []
            print0("disabling wandb.")
            os.environ["WANDB_MODE"] = "disabled"
            import wandb
            wandb.init(mode="disabled")
        ############################################################################
        num_devices = max(1, torch.cuda.device_count())
        num_cpus = count_cpus()
        num_workers = int(num_cpus / num_devices)
        if self.tiny_debug:
            print0("[tiny_debug] Setting num workers to 0")
            num_workers = 0
        self.eval_steps = int(self.eval_steps / num_devices * 512 / self.per_device_train_batch_size)
        self.save_steps = int(math.ceil(self.save_steps / num_devices * 512 / self.per_device_train_batch_size))
        self.warmup_steps = int(self.warmup_steps / num_devices)
        print0(f"training with eval_steps = {self.eval_steps} / num_workers = {num_workers} / warmup_steps = {self.warmup_steps} / save steps = {self.save_steps} / world_size {get_world_size()}")
        ############################################################################
        self.dataloader_num_workers = num_workers
        self.dataloader_persistent_workers = (num_workers > 0) # This may have been giving me weird deadlocks.
        self.dataloader_prefetch_factor = 4 if (num_workers > 0) else None
        self.dataloader_pin_memory = True
        today_date = datetime.date.today()
        formatted_date = today_date.strftime("%Y-%m-%d")
        if not self.exp_name.startswith("2024"):
            self.exp_name = f"{formatted_date}-{self.exp_name}"
        self.output_dir = os.path.join("saves", self.exp_name)
        logging.info(f"outputting model to directory: {self.output_dir}")
        logging.info(f"setting dataloader_drop_last from {self.dataloader_drop_last} -> {True}")
        self.dataloader_drop_last = True
        self.ddp_broadcast_buffers = False
        ############################################################################
        # self.metric_for_best_model = "loss"
        self.greater_is_better = False
        ############################################################################
        if self.wandb_exit_if_running_or_finished:
            exit_if_running_or_finished_wandb(
                project_name="tti-nomic-7",
                exp_name=self.exp_name,
                exp_group=self.exp_group
            )
