from typing import Optional

from dataclasses import dataclass, field

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
        default="sentence-transformers/gtr-t5-base",
        metadata={"help": "embedder name"}
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default="BeIR/nq", metadata={"help": "The name of the dataset to use (via the datasets library)."}
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
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
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
    steps_per_epoch: int = field(
        default=500_000, 
        metadata={
            "required": False,
            "help": "Size of pseudo-training set."
        },
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
    use_wandb: Optional[bool] = field(
        default=None, metadata={"help": "Whether or not to log to Weights & Biases."}
    )
    report_to: str = "wandb"

    save_total_limit: int = 1  # Maximum number of checkpoints to save.

    ##################### Experimental Settings ####################
    # exp_name: str = field(
    #     default=None,
    #     metadata={
    #         "required": True,
    #         "help": "Which experiment to run",
    #         "choices": ["prompt_tune", "fine_tune", "weighted_embeddings"],
    #     }
    # )

    def __setattr__(self, name, value):
        super(transformers.TrainingArguments, self).__setattr__(name, value)

    def __post_init__(self):
        super().__post_init__()
        ############################################################################
        # num_workers =  0
        # self.report_to = []
        ############################################################################
        num_workers = int(len(os.sched_getaffinity(0)) / torch.cuda.device_count())
        os.environ["RAYON_RS_NUM_CPUS"] = str(
            num_workers
        )  # Sets threads for hf tokenizers
        self.dataloader_num_workers = num_workers