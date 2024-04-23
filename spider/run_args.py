from typing import Optional

from dataclasses import dataclass, field

import datetime
import logging
import os

import torch
import transformers


from spider.lib import get_rank, get_world_size


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
        metadata={"help": "embedder name for the model (encoder-decoder)"}
    )
    embedder_rerank: str = field(
        default="sentence-transformers/gtr-t5-base",
        metadata={"help": "embedder name for reranking"}
    )
    architecture: str = field(
        default="biencoder",
        metadata = {
            "choices": ["biencoder", "query_independent_dt", "transductive", "two_head_mlp"],
        }
    )
    limit_layers: Optional[int] = field(
        default=None,
        metadata={
            "help": "If set, will load backbone and embedders with limited number of layers"
        }
    )
    gamma: float = field(
        default=0.9,
        metadata={
            "help": "Weighting between document and dataset embedding for dataset transformer"
        }
    )
    logit_scale: float = field(
        default=20,
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


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset: str = field(
        default="nomic_supervised", 
        metadata={
            "help": "The name of the dataset to use:",
            "choices": ["synthetic_words", "nomic_supervised", "nomic_unsupervised"]
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
        default="bm25",
        metadata={
            "help": "Model to use for clustering",
            "choices": ["bm25", "gtr_base"],
        }
    )
    clustering_query_to_doc: bool = field(
        default=False,
        metadata={
            "help": "Whether to use query-doc weightings for clustering, or just queries",
            "choices": [True, False],
        }
    )
    num_hard_negatives: int = field(
        default=0,
        metadata={
            "help": "Number of hard negatives for training",
        }
    )
    train_cluster_size: int = field(
        default=224,
        metadata={
            "help": "Cluster size for train data",
        }
    )
    eval_cluster_size: int = field(
        default=224,
        metadata={
            "help": "Cluster size for val data",
        }
    )
    use_prefix: bool = field(
        default=False,
        metadata={
            "help": "Whether to add domain-specific prefixes to data"
        }
    )
    def __post_init__(self):
        pass

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    dataset_info: str = field(
        default="fake",
        metadata={
            "help": "whether to use fake info for dataset (as opposed to our method)",
            "choices": ["random", "batch", "fake"],
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
    learning_rate: float = field(
        default=2e-5,
        metadata={"help": "The initial learning rate for AdamW on the backbone model."}
    )
    use_wandb: bool = field(
        default=False, metadata={"help": "Whether or not to log to Weights & Biases."}
    )
    report_to: str = "wandb"
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
        default=224,
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
        metadata={"help": "Max batch size for contrastive learning"}
    )
    eval_rerank_topk: int = field(
        default=128,
         metadata={"help": "Number of reranked examples during eval"}
    )
    save_strategy: str = "steps"
    save_steps: int = 4_000
    save_total_limit: int = 1  # Maximum number of checkpoints to save.

    exp_name: str = field(
        default=None,
        metadata={
            "help": "Name for this experiment, unique string",
            "required": "True",
        }
    )
    lr_scheduler_type: str = "constant_with_warmup"
    warmup_steps: int = field(
        default=8_000,
        metadata={
            "help": "Linear warmup over warmup_steps."
        }
    )
    weight_decay: float = field(
        default=0.01, 
        metadata={"help": "Weight decay for AdamW if we apply some."}
    )
    automatically_deduplicate_documents: bool = field(
        default=True,
        metadata={"help": "whether to count duplicate queries as joint positive samples"}
    )
    automatically_deduplicate_queries: bool = field(
        default=True,
        metadata={"help": "whether to count duplicate queries as joint positive samples"}
    )
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
        default=True,
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
    # dataloader_prefetch_factor: Optional[int] = field(
    #     default=4,
    #     metadata={
    #         "help": (
    #             "Number of batches loaded in advance by each worker. "
    #             "2 means there will be a total of 2 * num_workers batches prefetched across all workers. "
    #             "Default is 2 for PyTorch < 2.0.0 and otherwise None."
    #         )
    #     },
    # )
    def __setattr__(self, name, value):
        super(transformers.TrainingArguments, self).__setattr__(name, value)

    def __post_init__(self):
        super().__post_init__()
        ############################################################################
        if self.use_wandb:
            self.report_to = ["wandb"] if (self.local_rank <= 0) else []
        else:
            self.report_to = []
            print("disabling wandb.")
            os.environ["WANDB_MODE"] = "disabled"
        ############################################################################
        num_devices = max(1, torch.cuda.device_count())
        num_cpus = min(64, num_devices * 4, len(os.sched_getaffinity(0)))
        num_workers = int(num_cpus / num_devices)
        if self.tiny_debug:
            print("[tiny_debug] Setting num workers to 0")
            num_workers = 0
        self.eval_steps = int(self.eval_steps / num_devices)
        self.save_steps = int(self.save_steps / num_devices)
        self.warmup_steps = int(self.warmup_steps / num_devices)
        if get_rank() == 0:
            print(f"training with eval_steps = {self.eval_steps} / num_workers = {num_workers} / warmup_steps = {self.warmup_steps} / world_size {get_world_size()}")
        ############################################################################
        self.dataloader_num_workers = num_workers
        self.dataloader_persistent_workers = (num_workers > 0)
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
