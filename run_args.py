from typing import Optional

from dataclasses import dataclass, field

import datetime
import torch
import transformers

import os


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
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
        default=128,
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
        default="distilbert-base-uncased",
        metadata={"help": "embedder name for the model"}
    )
    dataset_backbone: str = field(
        default="distilbert-base-uncased",
        metadata={"help": "backbone model name"}
    )
    dataset_embedder: str = field(
        default="distilbert-base-uncased",
        metadata={"help": "embedder name for the model that embeds random dataset instances"}
    )
    embedder_rerank: str = field(
        default="sentence-transformers/gtr-t5-base",
        metadata={"help": "embedder name for reranking"}
    )
    architecture: str = field(
        default="query_dependent",
        metadata = {
            "choices": ["query_dependent", "query_independent", "biencoder_extended", "biencoder"],
        }
    )
    gamma: float = field(
        default=0.0,
        metadata={
            "help": ("weight (between 0 and 1) for residual term in document embeddings. "
                "0.0 means no residual, 1.0 means to ignore the second transformer block output."
            )
        }
    )
    limit_layers: Optional[int] = field(
        default=2, # None,
        metadata={
            "help": "If set, will load backbone and embedders with limited number of layers"
        }
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default="BeIR/nq", metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    num_hard_negatives: int = field(
        default=1, metadata={"help": "num hard negatives to use during training"}
    )
    dataset_config_name: Optional[str] = field(
        default="corpus", metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    def __post_init__(self):
        if self.dataset_name is None:
            raise ValueError("Need a dataset name.")

@dataclass
class TrainingArguments(transformers.TrainingArguments):
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
        default=100.0, 
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

    per_device_train_batch_size: int = field(
        default=64, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=64, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    evaluation_strategy: str = "steps"

    max_batch_size_fits_in_memory: int = field(
        default=64, 
        metadata={"help": "Max batch size for contrastive learning"}
    )

    eval_rerank_topk: int = field(
        default=100,
         metadata={"help": "Number of reranked examples during eval"}
    )
    save_total_limit: int = 1  # Maximum number of checkpoints to save.

    exp_name: str = field(
        default=None,
        metadata={
            "help": "Name for this experiment, unique string",
            "required": "True",
        }
    )
    def __setattr__(self, name, value):
        super(transformers.TrainingArguments, self).__setattr__(name, value)

    def __post_init__(self):
        super().__post_init__()
        if self.use_wandb:
            self.report_to = ["wandb"] if (self.local_rank <= 0) else []
        else:
            self.report_to = []
            print("disabling wandb.")
            os.environ["WANDB_MODE"] = "disabled"
        ############################################################################
        num_workers = int(len(os.sched_getaffinity(0)) / torch.cuda.device_count())
        # num_workers = 0 # For debugging
        ############################################################################
        os.environ["RAYON_RS_NUM_CPUS"] = str(
            num_workers
        )  # Sets threads for hf tokenizers
        self.dataloader_num_workers = num_workers
        self.dataloader_persistent_workers = True
        today_date = datetime.date.today()
        formatted_date = today_date.strftime("%Y-%m-%d")
        self.exp_name = f"{formatted_date}-{self.exp_name}"
        self.output_dir = os.path.join("saves", self.exp_name)
        print(f"outputting model to directory: {self.output_dir}")