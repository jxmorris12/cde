from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import functools
import gc
import math
import os
import re
import time

import datasets
import torch
import transformers
import wandb

from beir.retrieval.evaluation import EvaluateRetrieval
from transformers.trainer_utils import speed_metrics

from cde.dataset import BeirDataset
from cde.gradcache import GradCache
from cde.lib import (
    gather,
    get_rank,
    get_world_size,
    inputs_for_key,
    print0,
    verify_ddp_weights_equal,
    RerankHelper,
    TensorRunningAverages
)
from cde.lib.trainer_hn_filtering import TrainerNegativeFilterMixin
from cde.sampler import Sampler


def reset_memory() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def calculate_gradient_norm(model: torch.nn.Module) -> torch.Tensor:
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)  # Calculate the 2-norm of the gradients
            total_norm += param_norm.detach() ** 2
    total_norm = total_norm ** (1. / 2)  # Take the square root to get the total norm
    return total_norm


class CustomTrainer(transformers.Trainer, TrainerNegativeFilterMixin):
    retrieval_datasets: Dict[str, datasets.Dataset]
    dataset_backbone_tokenizer: transformers.PreTrainedTokenizer
    _train_sampler_fn: Callable[[], Sampler]
    _eval_sampler_fns: Callable[[], Dict[str, Sampler]]

    def __init__(
            self, 
            *args,
                dataset_backbone_tokenizer: transformers.PreTrainedTokenizer,
                retrieval_datasets: Dict[str, datasets.Dataset],
                train_sampler_fn: Callable[[], Sampler],
                eval_sampler_fns: Callable[[], Dict[str, Sampler]],
                data_args,
                model_args,
                **kwargs
            ):
        super().__init__(*args, **kwargs)
        self.max_seq_length = self.model.config.max_seq_length
        self.retrieval_datasets = retrieval_datasets
        self._train_sampler_fn = train_sampler_fn
        self._eval_sampler_fns = eval_sampler_fns
        self.data_args = data_args
        self.model_args = model_args
        self._signature_columns = [
            "idx",
            "query", "document",
            "document_input_ids", "document_attention_mask",
            "negative_document_input_ids", "negative_document_attention_mask",
            "dataset_input_ids", "dataset_attention_mask",
            "batch_dataset_input_ids", "batch_dataset_attention_mask",
            "random_document_input_ids", "random_document_attention_mask",
            "negative_document_input_ids", "negative_document_attention_mask",
            "query_input_ids", "query_attention_mask",
            "query_embedding", "document_embedding",
        ]
        self.dataset_backbone_tokenizer = dataset_backbone_tokenizer 
        self.use_gc = self.args.use_gc # whether to use gradcache
        self.gc = None # lazily initialized during training
        self._extra_logs = TensorRunningAverages()
        self.model_has_two_stages = hasattr(self.model, "second_stage_model")
        self._model_stages = None
        # Wrap first-stage and second-stage models in DDP/FSDP wrappers before all the HF trainer
        # boilerplate code is called.
        try:
            self.get_model_stages(model=self.model)
        except AttributeError:
            pass
        self._run_ddp_verify = False

        effective_batch_size = get_world_size() * self.args.per_device_train_batch_size
        self._memory_reset_step_frequency = int(2000 * effective_batch_size / 256)
        if (self.args.hn_tune_threshold is not None) and (self.args.hn_tune_threshold >= 0.0):
            self._init_hn_filter_model()
         
    
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        super()._save(output_dir=output_dir, state_dict=state_dict)
        
        if get_rank() == 0:
            # Save trainer data args and model args
            torch.save(
                self.data_args, os.path.join(output_dir, "data_args.bin")
            )
            torch.save(
               self.model_args, os.path.join(output_dir, "model_args.bin"),
            )
        
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Will save the model, so you can reload it using `from_pretrained()`.

        Will only save from the main process.
        """

        if output_dir is None:
            output_dir = self.args.output_dir
            
        if self.is_fsdp_enabled:
            if ("FULL_STATE_DICT" in str(self.accelerator.state.fsdp_plugin.state_dict_type)):
                # from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                # import torch.distributed.checkpoint.state_dict
                # with FSDP.summon_full_params(self.model):

                # https://github.com/pytorch/pytorch/issues/98823#issuecomment-1647496785
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                from torch.distributed.fsdp import StateDictType, FullStateDictConfig

                if self.model_has_two_stages:
                    # Weird FSDP hack: manually merge state dicts
                    first_stage_model, second_stage_model = self.get_model_stages(self.model)
                    save_policy = FullStateDictConfig(
                        offload_to_cpu=True, 
                        rank0_only=True,
                    )
                    with FSDP.state_dict_type(first_stage_model, StateDictType.FULL_STATE_DICT, save_policy):
                        first_stage_state_dict = first_stage_model.state_dict()
                        first_stage_state_dict = { f"first_stage_model.{k}": v for k, v in first_stage_state_dict.items() }
                    with FSDP.state_dict_type(second_stage_model, StateDictType.FULL_STATE_DICT, save_policy):
                        second_stage_state_dict = second_stage_model.state_dict()
                        second_stage_state_dict = { f"second_stage_model.{k}": v for k, v in second_stage_state_dict.items() }
                    cpu_state_dict = {**first_stage_state_dict, **second_stage_state_dict}

                    with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_policy):
                        if self.args.should_save:
                            self._save(output_dir, state_dict=cpu_state_dict)
                else:
                    with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_policy):
                        cpu_state_dict = self.model.state_dict()
                        if self.args.should_save:
                            self._save(output_dir, state_dict=cpu_state_dict)

        elif self.args.should_save:
            self._save(output_dir)

        # Push to the Hub when `save_model` is called by the user.
        if self.args.push_to_hub and not _internal_call:
            self.push_to_hub(commit_message="Model save")
    
    def consider_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.args.ddp_share_negatives_between_gpus:
            return gather(tensor)
        else:
            return tensor
    
    def _get_train_sampler(self) -> torch.utils.data.Sampler:
        return self._train_sampler_fn()
    
    def _get_eval_sampler(self, eval_dataset: datasets.Dataset) -> torch.utils.data.Sampler:
        return next(iter(self._eval_sampler_fns.values()))()
    
    def _log_extra(self, key: str, val: torch.Tensor):
        self._extra_logs.update(key, val)
    
    def log(self, logs: Dict[str, float]) -> None:
        """Override log to add other metrics we're tracking.
        """
        if self.is_in_train:
            logs.update(self._extra_logs.get_and_clear_all())
        super().log(logs)

    def create_optimizer(self, *args, **kwargs):
        super().create_optimizer(*args, **kwargs)

        if self.use_gc:
            gc_bs = self.args.max_batch_size_fits_in_memory
            gc_bs_fs = (
                self.args.max_batch_size_fits_in_memory_first_stage 
                or 
                self.args.max_batch_size_fits_in_memory
            )
            print0(f"[rank {get_rank()}] initializing GradCache with chunk_sizes={gc_bs} and {gc_bs_fs}.")
            self.gc = GradCache(
                accelerator=self.accelerator,
                chunk_sizes=[gc_bs, gc_bs],
                first_stage_chunk_sizes=[gc_bs_fs, gc_bs_fs],
                loss_fn=functools.partial(self._contrastive_loss, return_scores=False), 
                bf16=self.args.bf16,
            )
        else:
            self.gc = None
        
    def get_train_dataloader(self) -> torch.utils.data.DataLoader:
        sampler = self._get_train_sampler()
        sampler.set_epoch(self.state.epoch or 0)
        train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            collate_fn=self.data_collator,
            shuffle=False, # shuffling happens in the sampler
            batch_size=self.args.per_device_train_batch_size,
            drop_last=True,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            sampler=sampler
        )
        self.train_dataloader = train_dataloader
        return train_dataloader
        
    def get_eval_dataloader(self, 
                            eval_dataset: Optional[torch.utils.data.Dataset], 
                            sampler: Optional[Sampler] = None) -> torch.utils.data.DataLoader:
        """This is a clever bit of code that will do evaluation with
        a different dataset per evaluation batch.
        """
        data_collator = self.data_collator

        if isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")
        
        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            sampler=(sampler or self._get_eval_sampler(eval_dataset)),
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
        def advance_and_return(x):
            eval_dataloader.dataset.reset_dataset_idx()
            return x
        
        return map(advance_and_return, eval_dataloader)

    def _get_examples_table(self, dataloader, n: int = 100) -> wandb.Table:
        """Decodes column tensors back to strings and displays them
        in Weights & Biases.
        """
        if not self.args.use_wandb:
             return None
        elif not self._is_main_worker:
             return None
        batch = next(iter(dataloader))
        print("[get_examples_table] got batch", batch is None)
        if batch is None:
            return None
    
        keys = ["query_input_ids", "document_input_ids"]
        tokenizers = [self.dataset_backbone_tokenizer, self.dataset_backbone_tokenizer]

        bot_pattern = r"(<\|begin_of_text\|>)+"
        bot_replacement = "<|begin_of_text|>"
        data = [
                [re.sub(bot_pattern, bot_replacement, text) for text in tokenizer.batch_decode(batch[key][:n], skip_special_tokens=False)]
                for (tokenizer, key) in zip(tokenizers, keys)
        ]
        names = [k.replace("_input_ids", "") for k in keys]
        # transpose the list of lists
        data = list(map(list, zip(*data)))
        return wandb.Table(columns=names, data=data)

    @property
    def _is_main_worker(self) -> bool:
        return get_rank() == 0
    
    def _inner_training_loop(self, *args, **kwargs):
        """Override pre-training loop to do a couple of things. This happens after model is loaded from checkpoint."""
        # Log examples table!
        if self._is_main_worker:
            print0(f"[inner_training_loop] getting examples table")
            # On beginning of train, log tables of examples.
            train_table: wandb.Table = self._get_examples_table(
                super().get_train_dataloader()
            )
            if self.eval_dataset is not None:
                eval_table: wandb.Table = self._get_examples_table(
                    super().get_eval_dataloader()
                )
            else:
                eval_table = None
            wandb.log({
                "examples/train": train_table,
                "examples/eval": eval_table,
            })
        
        # Option to run eval at beginning of training
        run_eval_on_start_of_training = False
        if run_eval_on_start_of_training:
            self.evaluate_retrieval_datasets()

        super()._inner_training_loop(*args, **kwargs)
        
    def training_step(
            self, model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
        ) -> torch.Tensor:
        """
        Overriding from trainer so that we can disable backward()
        in favor of the gradcache backward().
        """
        self.train_dataloader.dataset.reset_dataset_idx()

        # Reset memory every once in awhile to try to help CPU/GPU OOM
        if (self.state.global_step > 0) and (self.state.global_step % self._memory_reset_step_frequency == 0):
            if get_rank() == 0:
                print("[rank 0] Resetting memory")
            reset_memory()

        model.train()
        inputs = self._prepare_inputs(inputs)
        # print("[1] training_step compute_loss", get_rank())
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        if self.args.n_gpu > 1:
            loss = loss.mean()
        # Uncomment next line to test eval straightaway.
        # self.control.should_evaluate = True  #########
        ##############################################
        if not self.use_gc:
            self.accelerator.backward(loss)
        self._log_grad_norm_metrics(model=model)
        if self._run_ddp_verify and (get_world_size() > 1) and self.state.global_step in [2, 2000, 20_000]:
            if get_rank() == 0: print("[verify_ddp_weights_equal] running at step", self.state.global_step)
            verify_ddp_weights_equal(model) # TODO: Only call this every once in a while
        # print("[3] training_step return", get_rank())
        return loss.detach() / self.args.gradient_accumulation_steps

    def _contrastive_loss(
                self, 
                e1: torch.Tensor, e2: torch.Tensor, 
                one_hot_labels: torch.Tensor,
                duplicate_labels: torch.Tensor,
                return_scores: bool = True
            ) -> Tuple[torch.Tensor, torch.Tensor]:
        e1 = self.consider_gather(e1)
        e2 = self.consider_gather(e2)

        assert len(e1.shape) == 2
        assert len(e2.shape) == 2
        assert e1.shape[1] == e2.shape[1]

        e1 = e1 / e1.norm(p=2, dim=1, keepdim=True) # Query
        e2 = e2 / e2.norm(p=2, dim=1, keepdim=True) # Document
        scores = e1 @ e2.T

        batch_size, _ = scores.shape
        # This multiply-by-20 thing is used in BeIR and LaPRador:
        # "When calculating the loss, we apply a re-scaling trick
        # of multiplying the cosine similarity score by 20 for
        # better optimization (Thakur et al., 2021)."
        # 
        scores *= self.model.temp
        assert scores.shape == one_hot_labels.shape, f"got different shapes {scores.shape} and {one_hot_labels.shape}"
        labels = one_hot_labels / one_hot_labels.sum(dim=1, keepdim=True)

        logits = scores - (duplicate_labels.float() * 10**10)
        loss_unscaled = torch.nn.functional.cross_entropy(
            logits, labels, label_smoothing=0.0
        ) 

        if self.args.ddp_share_negatives_between_gpus:
            # traditionally people multiply the contrastive loss by the number of gpus.
            # so we do it too here.
            loss = loss_unscaled * get_world_size()
        else:
            loss = loss_unscaled
        if (loss.isnan()):
            raise RuntimeError("Loss is nan!")
        
        pred_labels = one_hot_labels[torch.arange(len(one_hot_labels)), logits.argmax(dim=1)]
        acc = pred_labels.float().mean()

        # TODO: Gather loss and acc for logging.
        acc = self.consider_gather(acc.detach())
        loss_unscaled = self.consider_gather(loss_unscaled.detach())

        metrics = {
            "acc": acc,
            "loss_unscaled": loss_unscaled,
            "stats_emb_dim": e1.shape[-1],
            "stats_total_queries": len(e1),
            "stats_total_documents": len(e2),
            "batch_size": batch_size,
        }
        if self.is_in_train:
            for key, val in metrics.items():
                self._log_extra(key, val)
        
        if return_scores:
            return loss, scores
        else:
            return loss
    
    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`. Called during self.evalaute()
        """
        inputs = { key: value.to(self.args.device) for key, value in inputs.items() }
        with torch.no_grad():
            loss = self.compute_loss(model=model, inputs=inputs)

        logits, labels = None, None
        return loss, logits, labels

    def _split_inputs(self, inputs: Dict[str, torch.Tensor]) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Splits input and creates dataset inputs based on trainer settings.

        Args:
            inputs (Dict[str, torch.Tensor]) – all data inputs to model
        Returns:
            query_inputs
            document_inputs
            negative_document_inputs
            dataset_inputs
        """
        # print("inputs.keys() =>", inputs.keys())
        query_inputs = inputs_for_key(inputs, key="query")
        document_inputs = inputs_for_key(inputs, key="document")
        random_document_inputs = inputs_for_key(inputs, key="random_document")
        negative_document_inputs = inputs_for_key(inputs, key="negative_document")
        dataset_inputs = inputs_for_key(inputs, key="dataset")

        # Flatten hard negative ids
        if len(negative_document_inputs.get("input_ids", torch.tensor([])).shape) == 3:
            seq_length = negative_document_inputs["input_ids"].shape[2]
            negative_document_inputs["input_ids"] = negative_document_inputs["input_ids"].reshape(-1, seq_length)
            negative_document_inputs["attention_mask"] = negative_document_inputs["attention_mask"].reshape(-1, seq_length)
            
            if "input_ids_first_stage" in negative_document_inputs:
                negative_document_inputs["input_ids_first_stage"] = negative_document_inputs["input_ids_first_stage"].reshape(-1, seq_length)
                negative_document_inputs["attention_mask_first_stage"] = negative_document_inputs["attention_mask_first_stage"].reshape(-1, seq_length)

        batch_size = query_inputs["input_ids"].shape[0]
        if self.args.transductive_input_strategy == "fake":
            fake_seq_length = 128
            fake_dataset_input_ids = torch.ones(
                (batch_size, fake_seq_length), device=query_inputs["input_ids"].device,
                dtype=torch.long
            )
            fake_dataset_attention_mask = torch.ones(
                (batch_size, fake_seq_length), device=query_inputs["input_ids"].device,
                dtype=torch.long
            )
            dataset_inputs["input_ids"] = fake_dataset_input_ids
            dataset_inputs["attention_mask"] = fake_dataset_attention_mask
        elif self.args.transductive_input_strategy in ["topk", "random_corpus"]:
            if len(negative_document_inputs) and ("input_ids" in negative_document_inputs) and len(negative_document_inputs["input_ids"]):
                # Also consider negative documents in the contextual selection step
                dataset_inputs["input_ids"] = torch.cat(
                    (document_inputs.get("input_ids_first_stage", "input_ids"), negative_document_inputs.get("input_ids_first_stage", "input_ids")),
                    dim=0
                )
                dataset_inputs["attention_mask"] = torch.cat(
                    (document_inputs.get("attention_mask_first_stage", "attention_mask"), negative_document_inputs.get("attention_mask_first_stage", "attention_mask")),
                    dim=0
                )
            else:
                dataset_inputs["input_ids"] = document_inputs.get("input_ids_first_stage", "input_ids")
                dataset_inputs["attention_mask"] = document_inputs.get("attention_mask_first_stage", "attention_mask")

        elif self.args.transductive_input_strategy == "random_corpus":
            dataset_inputs["input_ids"] = random_document_inputs.get("input_ids_first_stage", "input_ids")
            dataset_inputs["attention_mask"] = random_document_inputs.get("attention_mask_first_stage", "attention_mask")
        else:
            pass
        
        # Aggregate all contextual inputs from all GPUs
        dataset_inputs["input_ids"] = self.consider_gather(dataset_inputs["input_ids"])
        dataset_inputs["attention_mask"] = self.consider_gather(dataset_inputs["attention_mask"])
    
        # Sample fewer inputs if batch size is too large
        effective_batch_size = len(dataset_inputs["input_ids"])
        transductive_corpus_size = (
            self.model.config.transductive_corpus_size * 
            self.model.config.transductive_tokens_per_document
        )
        # assert transductive_corpus_size <= effective_batch_size, "cannot provide more contextual inputs than in batch"

        # Randomly reorder dataset input ids.
        R1 = torch.randperm(effective_batch_size)[:transductive_corpus_size]
        R2 = torch.randperm(effective_batch_size)[:transductive_corpus_size]

        # Store on query 
        query_inputs["dataset_input_ids"] = dataset_inputs["input_ids"][R1]
        query_inputs["dataset_attention_mask"] = dataset_inputs["attention_mask"][R1]

        # Store on doc
        document_inputs["dataset_input_ids"] = dataset_inputs["input_ids"][R2]
        document_inputs["dataset_attention_mask"] = dataset_inputs["attention_mask"][R2]
        
        return (query_inputs, document_inputs, negative_document_inputs)

    def get_model_stages(self, model: torch.nn.Module) -> Optional[Tuple[torch.nn.Module, torch.nn.Module]]:
        """We have to individually wrap models when our model is two stages (in the contextual setting) so
        that we can make sure DDP works properly, since it needs to individually hook into the forward() of
        both models.
        """
        # TODO: Try doing this by wrapping self._wrap_model():
        #           https://github.com/huggingface/transformers/blame/main/src/transformers/trainer.py
        # TODO: properly set kwargs such as DDP broadcast buffers in the wrapped
        #           modules
        # print("rank", get_rank(), "wrapping model...")
        # model = self._wrap_model(model)
        if not self.model_has_two_stages:
            return None
        elif self._model_stages is None:
            # if self._is_main_worker:
            #     breakpoint()
            # torch.distributed.barrier()
            # We have to create separate DDP instances so that we can call the forward() functions individually
            # from the two halves of our model and have gradients sync properly.
            # if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = self.accelerator.unwrap_model(model)
            if hasattr(model, "module"):
                if self.use_gc:
                    self._model_stages = self.accelerator.prepare(
                        model.module.first_stage_model,
                        model.module.second_stage_model,
                    )
                else:
                    self._model_stages = [
                        model.module.first_stage_model,
                        model.module.second_stage_model,
                    ]
            else:
                # model.first_stage_model = self._wrap_model(model.first_stage_model)
                # model.second_stage_model = self._wrap_model(model.second_stage_model)
                if self.use_gc:
                    self._model_stages = self.accelerator.prepare(
                        model.first_stage_model, model.second_stage_model,
                    )
                else:
                    self._model_stages = [
                        model.first_stage_model,
                        model.second_stage_model,
                    ]
        return self._model_stages

    def _log_grad_norm_metrics(self, model: torch.nn.Module) -> None:
        if hasattr(model, "module"): 
            model = model.module
        if self.is_in_train:
            if self.model_has_two_stages:
                model_stages = self.get_model_stages(model=model)
                grad_norm_metrics = {
                    "grad_norm_first_stage": calculate_gradient_norm(model_stages[0]),
                    "grad_norm_second_stage": calculate_gradient_norm(model_stages[1]),
                }
            else:
                grad_norm_metrics = {
                    "grad_norm_only_stage": calculate_gradient_norm(self.model),
                }
            for key, val in grad_norm_metrics.items():
                self._log_extra(key, val)

    def compute_loss(
        self,
        model: transformers.PreTrainedModel,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
    ) -> Union[Tuple[torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor]:
        query_inputs, document_inputs, negative_document_inputs = self._split_inputs(inputs=inputs)

        # doc_random_integers = torch.randint(0, 2, (seq_len,))
        # query_random_integers = torch.randint(0, 2, (query_inputs["input_ids"].shape[1],))
        
        # doc_seq_len_hash = torch.where(random_integers == 0, torch.tensor(-1), torch.tensor(1)).long()
        # query_seq_len_hash = torch.where(random_integers == 0, torch.tensor(-1), torch.tensor(1)).long()

        # non_neg_doc_unique_ids = self.consider_gather(document_inputs["input_ids"]).cpu() @ doc_seq_len_hash
        if ("input_ids" in negative_document_inputs) and (negative_document_inputs["input_ids"].numel() > 0):
            document_inputs["input_ids"] = torch.cat(
                (document_inputs["input_ids"], negative_document_inputs["input_ids"]), dim=0
            )
            document_inputs["attention_mask"] = torch.cat(
                (document_inputs["attention_mask"], negative_document_inputs["attention_mask"]), dim=0
            )
            document_inputs["text"] = document_inputs["text"] + [hn_text for hn_text_list in negative_document_inputs["text"] for hn_text in hn_text_list]
            document_inputs["text__no_prefix"] = document_inputs["text__no_prefix"] + [hn_text for hn_text_list in negative_document_inputs["text__no_prefix"] for hn_text in hn_text_list]
            #########################################################################################################
            negative_document_inputs["dataset_input_ids"] = document_inputs["dataset_input_ids"]
            negative_document_inputs["dataset_attention_mask"] = document_inputs["dataset_attention_mask"]
        
        all_document_input_ids = self.consider_gather(document_inputs["input_ids"])
        all_query_input_ids = self.consider_gather(query_inputs["input_ids"])
        document_first_tokens = all_document_input_ids[:, 1].contiguous()
        query_first_tokens = all_query_input_ids[:, 1].contiguous()

        # Create labels based on document IDs.
        # document_unique_ids = all_document_input_ids.cpu() @ doc_seq_len_hash

        # We have to stack the hard negatives within device (before the gather) to mimic
        # the way that embeddings are computed and gathered.
        all_idx = inputs["idx"].flatten()
        num_hn = len(document_inputs["input_ids"]) - len(all_idx)
        if num_hn > 0:
            hn_idx = torch.zeros((num_hn,), dtype=torch.long, device=self.args.device) - 1
            all_idx = torch.cat((all_idx, hn_idx), dim=0)
            one_hot_labels = self.consider_gather(inputs["idx"].flatten())[:, None] == self.consider_gather(all_idx)[None, :]
        else:
            # doing this without a gather because that takes forever with large batches.
            batch_size = document_first_tokens.shape[0]
            one_hot_labels = torch.eye(batch_size, device=self.args.device, dtype=torch.bool)

        smart_labels = one_hot_labels.cpu().clone()
        # Automatically filter out too-hard negatives.
        hard_negatives_metrics = {}
        if (self.args.hn_tune_threshold is not None) and (self.args.hn_tune_threshold > 0.0):
            qd_scores = self._get_query_doc_scores(query_inputs, document_inputs)
            qd_scores = qd_scores.to(smart_labels.device)

            if self.args.hn_tune_threshold == 1.0:
                smart_labels_neg = qd_scores >= qd_scores.diag()[:, None]
            else:
                smart_labels_neg = qd_scores >= self.args.hn_tune_threshold
            qd_metrics = {
                "smart_hn_exceed_threshold": (qd_scores >= self.args.hn_tune_threshold).long().sum(),
                "smart_hn_score_mean": qd_scores.mean(),
                "smart_hn_score_max": qd_scores.max(),
                "smart_hn_score_min": qd_scores.min(),
            }
        else:
            smart_labels_neg = smart_labels
            qd_metrics = {}
            # Don't mask hard negatives for a given sample!
        if num_hn > 0:
            neg_doc_idx = torch.cat((inputs["idx"].flatten(), negative_document_inputs["idx"].flatten()))
            hn_match_mask = (inputs["idx"].flatten()[:, None] == neg_doc_idx[None, :]).to(smart_labels_neg.device)
            if self.args.use_in_batch_negatives:
                smart_labels_neg = smart_labels_neg & (~hn_match_mask)
            else:
                smart_labels_neg = smart_labels_neg | (~hn_match_mask)
        
        assert smart_labels_neg.shape == smart_labels.shape, f"got different shapes {smart_labels_neg.shape} and {smart_labels.shape}"

        smart_labels = smart_labels | smart_labels_neg
        hard_negatives_metrics = qd_metrics | {
            "smart_hn_neg_mean": smart_labels_neg.float().mean(),
        }

        # Aggregate labels for duplicate queries.
        # num_unique_documents = document_unique_ids.unique().numel()
        # num_collisions_documents = len(document_unique_ids) - num_unique_documents
        # num_unique_queries = query_unique_ids.unique().numel()
        # num_collisions_queries = len(smart_labels) - num_unique_queries

        smart_labels = smart_labels.to(self.args.device)
        duplicate_labels = (smart_labels.long() - one_hot_labels.long())

        # Dataset input stats
        ds_input_document_unique_tokens = document_inputs["dataset_input_ids"].unique().numel()

        query_first_token_most_common = query_first_tokens.flatten().mode().values.detach()
        query_first_token_mean = (query_first_tokens == query_first_token_most_common).float().mean()

        document_first_token_most_common = document_first_tokens.flatten().mode().values.detach()
        document_first_token_mean = (document_first_tokens == document_first_token_most_common).float().mean()

        metrics = {
            # "stats_unique": num_unique_documents,
            # "stats_unique_queries": num_unique_queries,
            # "stats_collisions": num_collisions_documents,
            # "stats_collisions_queries": num_collisions_queries,
            "stats_dataset_inputs_unique_tokens": ds_input_document_unique_tokens,
            "stats_unique_first_tokens_document": document_first_tokens.unique().numel(),
            "stats_unique_first_tokens_query": query_first_tokens.unique().numel(),
            "stats_num_ds_inputs": len(document_inputs["dataset_input_ids"]),
            ###############################################################################
            "stats_unique_first_tokens_document": document_first_tokens.unique().numel(),
            "stats_unique_first_tokens_query": query_first_tokens.unique().numel(),
            "stats_query_first_token_mean": query_first_token_mean,
            "stats_query_first_token_value_mean": query_first_tokens.float().mean(),
            "stats_document_first_token_mean": document_first_token_mean,
            "stats_document_first_token_value_mean": document_first_tokens.float().mean(),
            ###############################################################################
            "stats_one_hot_labels_sum": one_hot_labels.long().sum().detach(),
            "stats_smart_labels_sum": smart_labels.long().sum().detach(),
            "stats_smart_labels_shape_0": smart_labels.shape[0],
            "stats_smart_labels_shape_1": smart_labels.shape[1],
            **hard_negatives_metrics,
        }
        if self.is_in_train:
            for key, val in metrics.items():
                self._log_extra(key, val)

        if self.use_gc:
            # TODO: Restore gradcache w/ hard negatives.
            def empty_backward(_loss):
                pass
            backward_fn = (
                self.accelerator.backward if self.model.training else empty_backward
            )
            loss = self.gc(
                query_inputs, 
                document_inputs, 
                model=self._wrap_model(model),
                model_stages=self.get_model_stages(model=model),
                no_sync_except_last=(get_world_size() > 1),
                backward_fn=backward_fn,
                run_backward=self.model.training,
                ################################
                one_hot_labels=one_hot_labels,
                duplicate_labels=duplicate_labels,
            )
            return loss
        else:
            e1 = model(**query_inputs)
            e2 = model(**document_inputs)

            loss, _scores = self._contrastive_loss(
                e1, e2, 
                one_hot_labels=one_hot_labels,
                duplicate_labels=duplicate_labels,
            )
            return loss
    
    # Custom retrieval evalution code
    def _retrieval_evaluate(
            self,
            eval_dataset: BeirDataset,
            model: torch.nn.Module,
            metric_key_prefix: str,
            n: int,
        ) -> Dict[str, float]:
        if len(eval_dataset.queries) < get_world_size():
            print0("WARNING: less than world size queries -- skipping eval.")
            return {}

        # github.com/jxmorris12/tti/blob/master/trainer.py
        # github.com/beir-cellar/beir/blob/main/examples/retrieval/evaluation/reranking/evaluate_bm25_ce_reranking.py
        data_collator = self._get_collator_with_removed_columns(
            self.data_collator, description="evaluation")
        dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=self.args.eval_batch_size,
            collate_fn=data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
        model = self._wrap_model(model, training=False, dataloader=dataloader)

        model.eval()
        
        reranker = RerankHelper(
            model=model, 
            tokenizer=self.dataset_backbone_tokenizer, 
            max_seq_length=self.max_seq_length,
            batch_size=self.args.max_batch_size_fits_in_memory,
            max_reranking_queries=n,
            name=metric_key_prefix,
            transductive_n_outputs_ensemble=self.args.transductive_n_outputs_ensemble,
            transductive_input_strategy=self.args.transductive_input_strategy,
        )
        # Rerank top-k results using the reranker provided
        rerank_results_model = reranker.rerank(
            dataset=eval_dataset,
            top_k=self.args.eval_rerank_topk,
        )

        #### Evaluate your retrieval using NDCG@k, MAP@K ...
        ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(
            eval_dataset.qrels, rerank_results_model, [1, 5, 10, 100]
        )
        retrieval_metrics = {
            **ndcg, **_map, **recall, **precision
        }
        # Take the average scores of things. We're hoping this will give us a less noisy metric.
        weighted_scores = []
        for query_key, query_scores in rerank_results_model.items():
            for true_doc_key, true_doc_weight in eval_dataset.qrels[query_key].items():
                weighted_scores.append(
                    query_scores.get(true_doc_key, 0.0) * true_doc_weight
                )
        retrieval_metrics["score_weighted_qrels/mean"] = sum(weighted_scores) / len(weighted_scores)
        retrieval_metrics["score_weighted_qrels/sum"] = sum(weighted_scores)
        retrieval_metrics["score_weighted_qrels/total"] = len(weighted_scores)

        return retrieval_metrics

    def evaluate(
        self,
        eval_dataset: Optional[Union[datasets.Dataset, Dict[str, datasets.Dataset]]] = None,
        **kwargs
    ) -> Dict[str, float]:
        if eval_dataset is None and (self.eval_dataset is None):
            # "Skipping eval – no eval dataset found"
            return {}
        else:
            return super().evaluate(
                eval_dataset=eval_dataset,
                **kwargs
            )
    
    def _run_eval_dataloader(
            self,
            eval_dataloader: torch.utils.data.DataLoader,
            metric_key_prefix: str,
            ignore_keys: Optional[List[str]] = None,
        ) -> Dict[str, float]:
        self._memory_tracker.start()
        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)
        return output.metrics
        
    def evaluate(
        self,
        eval_dataset: Optional[Union[datasets.Dataset, Dict[str, datasets.Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        # handle multipe eval datasets
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if isinstance(eval_dataset, dict):
            metrics = {}
            for eval_dataset_name, _eval_dataset in eval_dataset.items():
                dataset_metrics = self.evaluate(
                    eval_dataset=_eval_dataset,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=f"{metric_key_prefix}_{eval_dataset_name}",
                )
                metrics.update(dataset_metrics)
            return metrics
        elif eval_dataset is None:
            return {}

        # aggregate metrics over multiple samplers
        all_metrics = {}
        eval_samplers = { name: fn() for name, fn in self._eval_sampler_fns.items() }
        for sampler_name, sampler in eval_samplers.items():
            eval_with_sampler_key = "_".join([metric_key_prefix, sampler_name])
            eval_dataloader = self.get_eval_dataloader(eval_dataset, sampler=sampler)
            metrics = self._run_eval_dataloader(
                eval_dataloader=eval_dataloader,
                ignore_keys=eval_dataloader,
                metric_key_prefix=eval_with_sampler_key
            )
            all_metrics = (metrics | all_metrics)
            reset_memory()
        return all_metrics

    def evaluate_retrieval_datasets(self) -> Dict[str, float]:
        model = self.model
        all_metrics = {}
        n = self.args.num_eval_rerank_samples
        for eval_dataset_name, eval_dataset in self.retrieval_datasets.items():
            metric_key_prefix = f"eval_{eval_dataset_name}"
            metrics = self._retrieval_evaluate(
                eval_dataset=eval_dataset,
                model=model,
                metric_key_prefix=metric_key_prefix,
                n=n,
            )
            # Prefix all keys with metric_key_prefix + '_'
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
                    
            self._memory_tracker.stop_and_update_metrics(metrics)
            self.log(metrics)
            all_metrics.update(metrics)

        if len(all_metrics) and get_rank() == 0:
            ndcg_str = "NDCG@10"
            ndcg_metrics = { k: v for k,v in all_metrics.items() if k.endswith(ndcg_str) }
            print("evaluate_retrieval_datasets =>", ndcg_metrics)
            all_metrics["NDCG@10/sum"] = sum(ndcg_metrics.values())
            all_metrics["NDCG@10/mean"] =  sum(ndcg_metrics.values()) / len(ndcg_metrics)
        return all_metrics

    def _maybe_log_save_evaluate(self, tr_loss: float, grad_norm: float, model: torch.nn.Module, *args, **kwargs):
        ####   (have to hook in here to call my special function)
        # 
        # evaluate on retrieval datasets!

        # check should_evaluate bool because it'll be reset by call to super()..
        should_evaluate = self.control.should_evaluate
        
        # do all the other stuff
        super()._maybe_log_save_evaluate(tr_loss, grad_norm, model, *args, **kwargs)

        # do my custom eval
        if should_evaluate:
            self.evaluate_retrieval_datasets()
        reset_memory()
