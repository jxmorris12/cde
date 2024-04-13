from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import functools
import gc
import math
import os
import time

import datasets
import torch
from transformers.trainer_utils import speed_metrics
import transformers
import wandb

from beir.retrieval.evaluation import EvaluateRetrieval
from gradcache import GradCache
from dataset import BeirDataset
from lib import gather, get_rank, get_world_size, inputs_for_key, RerankHelper, TensorRunningAverages
from sampler import Sampler


def calculate_gradient_norm(model: torch.nn.Module):
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)  # Calculate the 2-norm of the gradients
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)  # Take the square root to get the total norm
    return total_norm


class CustomTrainer(transformers.Trainer):
    retrieval_datasets: Dict[str, datasets.Dataset]
    embedder_tokenizer: transformers.PreTrainedTokenizer
    dataset_tokenizer:  transformers.PreTrainedTokenizer
    _train_sampler_fn: Callable[[], Sampler]
    _eval_sampler_fns: Callable[[], Dict[str, Sampler]]

    def __init__(self, *args,
                 embedder_tokenizer: transformers.PreTrainedTokenizer,
                 dataset_tokenizer:  transformers.PreTrainedTokenizer,
                 retrieval_datasets: Dict[str, datasets.Dataset],
                 train_sampler_fn: Callable[[], Sampler],
                 eval_sampler_fns: Callable[[], Dict[str, Sampler]],
                  **kwargs
                ):
        super().__init__(*args, **kwargs)
        self.max_seq_length = self.model.config.max_seq_length
        self.retrieval_datasets = retrieval_datasets
        self._train_sampler_fn = train_sampler_fn
        self._eval_sampler_fns = eval_sampler_fns
        self._signature_columns = [
            "idx",
            "query", "document",
            "document_input_ids", "document_attention_mask",
            "negative_document_input_ids", "negative_document_attention_mask",
            "dataset_input_ids", "dataset_attention_mask",
            "batch_dataset_input_ids", "batch_dataset_attention_mask",
            "batch_dataset_input_ids", "batch_dataset_attention_mask",
            "random_document_input_ids", "random_document_attention_mask",
            "negative_document_input_ids", "negative_document_attention_mask",
            "query_input_ids", "query_attention_mask",
        ]
        self.embedder_tokenizer = embedder_tokenizer 
        self.dataset_tokenizer = dataset_tokenizer
        self.use_gc = self.args.use_gc # whether to use gradcache
        self.gc = None # lazily initialized during training
        self._extra_logs = TensorRunningAverages()
    
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
            print(f"[rank {get_rank()}] initializing GradCache with chunk_size={gc_bs}.")
            self.gc = GradCache(
                accelerator=self.accelerator,
                chunk_sizes=[gc_bs, gc_bs],
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
            drop_last=self.args.dataloader_drop_last,
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

    def _get_examples_table(self, dataloader, n: int = 64) -> wandb.Table:
        """Decodes column tensors back to strings and displays them
        in Weights & Biases.
        """
        if not self.args.use_wandb:
             return None
        elif not (self.args.local_rank <= 0):
             return None
        batch = next(iter(dataloader))
        if batch is None:
            return None
    
        keys = ["query_input_ids", "document_input_ids"]
        tokenizers = [ self.embedder_tokenizer,  self.embedder_tokenizer, self.dataset_tokenizer]
        data = [
                tokenizer.batch_decode(batch[key][:n], skip_special_tokens=True)
                for (tokenizer, key) in zip(tokenizers, keys)

        ]
        names = [k.replace("_input_ids", "") for k in keys]
        # transpose the list of lists
        data = list(map(list, zip(*data)))
        return wandb.Table(columns=names, data=data)

    @property
    def _is_main_worker(self) -> bool:
        return (
            (self.args.local_rank <= 0) and 
            (int(os.environ.get("LOCAL_RANK", 0)) <= 0)
        )
    
    def _inner_training_loop(self, *args, **kwargs):
        """Override pre-training loop to do a couple of things. This happens after model is loaded from checkpoint."""
        # Log examples table!
        if self._is_main_worker:
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
        
    def training_step(self, model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        # Reset dataloader index
        self.train_dataloader.dataset.reset_dataset_idx()

        if not self.use_gc:
            return super().training_step(model=model, inputs=inputs)

        # Overriding from trainer so that we can disable backward()
        # in favor of the gradcache backward()
        model.train()
        inputs = self._prepare_inputs(inputs)
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        if self.args.n_gpu > 1:
            loss = loss.mean()
        # Uncomment next line to test eval straightaway.
        # self.control.should_evaluate = True  #########
        ##############################################
        return loss.detach() / self.args.gradient_accumulation_steps

    def _contrastive_loss(
                self, 
                e1: torch.Tensor, e2: torch.Tensor, 
                one_hot_labels: torch.Tensor,
                duplicate_labels: torch.Tensor,
                return_scores: bool = True
            ) -> Tuple[torch.Tensor, torch.Tensor]:
        e1 = gather(e1)
        e2 = gather(e2)

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
        loss = torch.nn.functional.cross_entropy(
            logits, labels, label_smoothing=0.0
        ) * get_world_size()
        if (loss.isnan()):
            raise RuntimeError("Loss is nan!")
        
        pred_labels = one_hot_labels[torch.arange(len(one_hot_labels)), logits.argmax(dim=1)]
        acc = pred_labels.float().mean()

        metrics = {
            "acc": acc.item(),
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
        query_inputs = inputs_for_key(inputs, key="query")
        document_inputs = inputs_for_key(inputs, key="document")
        random_document_inputs = inputs_for_key(inputs, key="random_document")
        negative_document_inputs = inputs_for_key(inputs, key="negative_document")
        dataset_inputs = inputs_for_key(inputs, key="dataset")

        batch_size = query_inputs["input_ids"].shape[0]
        if self.args.dataset_info == "fake":
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
        elif self.args.dataset_info == "batch":
            dataset_inputs["input_ids"] = document_inputs["input_ids"]
            dataset_inputs["attention_mask"] = document_inputs["attention_mask"]
        elif self.args.dataset_info == "random":
            dataset_inputs["input_ids"] = random_document_inputs["input_ids"]
            dataset_inputs["attention_mask"] = random_document_inputs["attention_mask"]
        else:
            pass
        
        # if get_world_size() > 1:
        #     # Aggregate all transductive inputs from all GPUs
        #     dataset_inputs["input_ids"] = gather(dataset_inputs["input_ids"])
        #     dataset_inputs["attention_mask"] = gather(dataset_inputs["attention_mask"])
    
        # # Sample fewer inputs if batch size is too large
        # effective_batch_size = len(dataset_inputs["input_ids"])
        transductive_corpus_size = self.args.transductive_corpus_size
        # assert transductive_corpus_size <= effective_batch_size, "cannot provide more transductive inputs than in batch"
        # if transductive_corpus_size < effective_batch_size:
        #     C_perm = torch.randperm(effective_batch_size, device=query_inputs["input_ids"].device)
        #     C_perm = C_perm[:transductive_corpus_size]
        #     # Take the random indices from worker 0
        #     if get_world_size() > 1:
        #         torch.distributed.broadcast(C_perm, src=0)

        #     dataset_inputs["input_ids"] = dataset_inputs["input_ids"][C_perm]
        #     dataset_inputs["attention_mask"] = dataset_inputs["attention_mask"][C_perm]

        # Randomly reorder dataset input ids.
        R1 = torch.randperm(transductive_corpus_size)
        R2 = torch.randperm(transductive_corpus_size)

        # Store on query 
        query_inputs["dataset_input_ids"] = dataset_inputs["input_ids"][R1]
        query_inputs["dataset_attention_mask"] = dataset_inputs["attention_mask"][R1]

        # Store on doc
        document_inputs["dataset_input_ids"] = dataset_inputs["input_ids"][R2]
        document_inputs["dataset_attention_mask"] = dataset_inputs["attention_mask"][R2]

        return (query_inputs, document_inputs, negative_document_inputs)

    def compute_loss(
        self,
        model: transformers.PreTrainedModel,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
    ) -> Union[Tuple[torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor]:
        query_inputs, document_inputs, negative_document_inputs = self._split_inputs(inputs=inputs)
        
        # Uncomment next line to log stuff on every GPU.
        # print(f"[rank {get_rank()}] query 0 =", self.embedder_tokenizer.decode(document_inputs["input_ids"][0], skip_special_tokens=True))

        # Create labels based on document IDs.
        document_unique_ids = gather(document_inputs["input_ids"].sum(dim=1))
        idx = gather(inputs["idx"])
        one_hot_labels = (idx[:, None] == idx[None, :]).float()
        if self.args.automatically_deduplicate_documents:
            smart_labels = (
                document_unique_ids[:, None] == document_unique_ids[None, :])
        else:
            smart_labels = one_hot_labels.clone()
        
        # Aggregate labels for duplicate queries.
        query_unique_ids = gather(query_inputs["input_ids"].sum(dim=1))
        if self.args.automatically_deduplicate_queries:
            smart_labels = (
                (query_unique_ids[:, None] == query_unique_ids[None, :]).float() @ smart_labels.float())
            smart_labels = smart_labels.bool()

        num_unique_documents = document_unique_ids.unique().numel()
        num_collisions_documents = len(document_unique_ids) - num_unique_documents
        num_unique_queries = query_unique_ids.unique().numel()
        num_collisions_queries = len(smart_labels) - num_unique_queries

        duplicate_labels = (smart_labels.long() - one_hot_labels.long())

        # Dataset input stats
        ds_input_document_unique_tokens = document_inputs["dataset_input_ids"].unique().numel()

        metrics = {
            "stats_unique": num_unique_documents,
            "stats_unique_queries": num_unique_queries,
            "stats_collisions": num_collisions_documents,
            "stats_collisions_queries": num_collisions_queries,
            "stats_dataset_inputs_unique_tokens": ds_input_document_unique_tokens,
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
                model=model,
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

            if len(negative_document_inputs):
                negative_document_inputs["dataset_input_ids"] = document_inputs["dataset_input_ids"]
                negative_document_inputs["dataset_attention_mask"] = document_inputs["dataset_attention_mask"]
                e2_hn = model(**negative_document_inputs)
                e2 = torch.cat((e2, e2_hn), dim=0)
                negative_document_unique_ids = negative_document_inputs["input_ids"].sum(dim=1)
                corpus_unique_ids = torch.cat(
                    (corpus_unique_ids, negative_document_unique_ids), dim=0
                )

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
        ) -> Dict[str, float]:
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
            tokenizer=self.embedder_tokenizer, 
            batch_size=self.args.max_batch_size_fits_in_memory,
            max_seq_length=self.max_seq_length,
            name=metric_key_prefix,
            fake_dataset_info=(self.args.dataset_info == "fake"),
        )

        # Rerank top-k results using the reranker provided
        rerank_device = ("cuda" if torch.cuda.is_available() else "cpu")
        with torch.autocast(rerank_device, dtype=torch.bfloat16):
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
        return all_metrics

    def evaluate_retrieval_datasets(self) -> Dict[str, float]:
        model = self.model
        all_metrics = {}
        for eval_dataset_name, eval_dataset in self.retrieval_datasets.items():
            metric_key_prefix = f"eval_{eval_dataset_name}"
            metrics = self._retrieval_evaluate(
                eval_dataset=eval_dataset,
                model=model,
                metric_key_prefix=metric_key_prefix,
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
            # TODO: implement multi-gpu retrieval evaluation.
            self.evaluate_retrieval_datasets()
            gc.collect()
