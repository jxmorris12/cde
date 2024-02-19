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
        default="t5-base",
        metadata={"help": "embedder name for the model (encoder-decoder)"}
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

@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset: Optional[str] = field(
        default="reddit_supervised", metadata={
            "help": "The name of the dataset to use:",
            "choices": ["synthetic_words", "reddit_supervised", "reddit_unsupervised"]
        }
    )
    def __post_init__(self):
        pass

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    dataset_info: str = field(
        default="random",
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
    save_strategy: str = "steps"
    save_steps: int = 5000
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
        default=2_000,
        metadata={
            "help": "Linear warmup over warmup_steps."
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
        # num_workers = 1 # For debugging
        ############################################################################
        # os.environ["RAYON_RS_NUM_CPUS"] = str(
        #     num_workers
        # )  # Sets threads for hf tokenizers
        self.dataloader_num_workers = num_workers
        self.dataloader_persistent_workers = (num_workers > 0)
        self.dataloader_persistent_workers = False # Disabling to see if this fixes an error I had
        self.dataloader_pin_memory = True
        today_date = datetime.date.today()
        formatted_date = today_date.strftime("%Y-%m-%d")
        self.exp_name__no_date = "self.exp_name"
        self.exp_name = f"{formatted_date}-{self.exp_name}"
        self.output_dir = os.path.join("saves", self.exp_name)
        print(f"outputting model to directory: {self.output_dir}")
        print(f"setting dataloader_drop_last from {self.dataloader_drop_last} -> {True}")
        self.dataloader_drop_last = True
