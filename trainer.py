from typing import Any, Dict, Optional, Tuple, Union

import datasets
import torch
import transformers

import grad_cache
import wandb

from dataset import BeirDataset
from helpers import RerankHelper
from model import Model

def inputs_for_key(inputs: Dict[str, torch.Tensor], key: str):
    key += "_"
    return {k.replace(key, ""): v for k,v in inputs.items() if k.startswith(key)}


class CustomTrainer(transformers.Trainer):
    retrieval_datasets: Dict[str, datasets.Dataset]

    def __init__(self, *args,
                 retrieval_datasets: Dict[str, datasets.Dataset], **kwargs):
        super().__init__(*args, **kwargs)
        self.retrieval_datasets = retrieval_datasets
        self._signature_columns = [
            "idx", "query_embedding", "document_embeddings", "negative_document_embeddings",
            "document_input_ids", "document_attention_mask",
            "negative_document_input_ids", "negative_document_attention_mask",
            "query_input_ids", "query_attention_mask",
        ]
        print(f"initializing GradCache with chunk_size={self.args.max_batch_size_fits_in_memory}.")
        self.gc = grad_cache.GradCache(
            models=[self.model.forward_embedder, self.model.forward_embedder],
            chunk_sizes=self.args.max_batch_size_fits_in_memory,
            loss_fn=self._contrastive_loss, 
            fp16=self.args.fp16,
            scaler=self.scaler if self.args.fp16 else None,
        )
    
    def evaluate(*args, **kwargs) -> Dict[str, float]:
        return {}
    
    def training_step(self, model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        # Overriding from trainer so that we can disable backward()
        # in favor of the gradcache backward()
        model.train()
        inputs = self._prepare_inputs(inputs)
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        if self.args.n_gpu > 1:
            loss = loss.mean()
        return loss.detach() / self.args.gradient_accumulation_steps

    def _contrastive_loss(self, e1: torch.Tensor, e2: torch.Tensor) -> torch.Tensor:
        batch_size = len(e1)
        corpus_size = len(e2)

        scores = self.model(
            query_embedding=e1,
            document_embeddings=e2,
        )
        # This multiply-by-20 thing is used in BeIR and LaPRador:
        # "When calculating the loss, we apply a re-scaling trick
        # of multiplying the cosine similarity score by 20 for
        # better optimization (Thakur et al., 2021)."
        # 
        # scores *= 20  # TODO argparse: self.args.contrastive_temperature.exp()
        diagonal_idxs = torch.arange(batch_size, device=e1.device)
        import pdb; pdb.set_trace()
        loss = torch.nn.functional.cross_entropy(
            scores, diagonal_idxs, label_smoothing=0.0
        )
        if (loss.isnan()):
            raise RuntimeError("Loss is nan!")
        
        labels = torch.arange(batch_size, dtype=torch.long, device=scores.device)
        original_acc = ((
            torch.nn.functional.cosine_similarity(e1[:, None], e2[None, :], dim=2)
        ).argmax(1) == labels).float().mean()
        new_acc = (scores.argmax(1) == labels).float().mean()
        wandb.log({
            "train/acc_emb": original_acc.item(),
            "train/acc": new_acc.item(),
            "corpus_size": corpus_size,
        })

        # print(f"acc: {original_acc.item()*100:.1f} / now: {new_acc.item()*100:.1f}")
        # TODO restore this :)
        # smart_labels = (inputs["idx"][:, None] == inputs["idx"]).float()
        # smart_labels /= smart_labels.sum(dim=1)
        # zero_labels = torch.zeros((batch_size, len(inputs["negative_document_embeddings"])), dtype=torch.float32, device=scores.device)
        # smart_labels = torch.cat((smart_labels, zero_labels), dim=1)
                
        return loss

    def compute_loss(
        self,
        model: Model,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
    ) -> Union[Tuple[torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor]:
        # refer to jack's prev implementation:
        #       https://github.com/jxmorris12/tti/blob/master/trainer.py
        batch_size = inputs["query_embedding"].shape[0]

        # if self.model.config.freeze_embedder:
        #     query_embedding = inputs["query_embedding"]
        #     document_embeddings = inputs["document_embeddings"]
        #     negative_document_embeddings = inputs["negative_document_embeddings"]

        query_inputs = inputs_for_key(inputs, key="query")
        document_inputs = inputs_for_key(inputs, key="document")
        negative_document_inputs = inputs_for_key(inputs, key="negative_document")

        all_document_inputs = {
            k: (
                torch.cat((document_inputs[k], negative_document_inputs[k]), dim=0)
            )
            for k in ["input_ids", "attention_mask"]
        }
        # print("query_inputs >>", {k: v.shape for k,v in query_inputs.items()})
        # print("all_document_inputs >>", {k: v.shape for k,v in all_document_inputs.items()})
        return self.gc(query_inputs, all_document_inputs, no_sync_except_last=False)
    
    # Custom retrieval evalution code
    def _retrieval_evaluate(
            self,
            eval_dataset: BeirDataset,
            metric_key_prefix: str,
        ) -> Dict[str, float]:
        # github.com/jxmorris12/tti/blob/master/trainer.py
        # github.com/beir-cellar/beir/blob/main/examples/retrieval/evaluation/reranking/evaluate_bm25_ce_reranking.py
        from beir.retrieval.evaluation import EvaluateRetrieval

        dataloader = self.get_eval_dataloader(eval_dataset)
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        # if full fp16 or bf16 eval is wanted and this funciton isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if self.args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=self.args.device)
            elif self.args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=self.args.device)
        model.eval()
        
        reranker = RerankHelper(self.model)

        # TODO cache this if it's slow
        # index_name = ""
        # hostname = "rush-compute-01"
        # username = "elastic"
        # password = "FjZD_LI-=AJOtsfpq9U*"

        # url = f"https://{username}:{password}@rush-compute-01.tech.cornell.edu:9200""

        # Rerank top-100 results using the reranker provided
        rerank_results_model, rerank_results_biencoder = reranker.rerank(
            eval_dataset.corpus, 
            eval_dataset.corpus_embeddings,
            eval_dataset.queries, 
            eval_dataset.query_embeddings,
            results=eval_dataset.ance_results, top_k=self.args.per_device_train_batch_size * 2
        )

        #### Evaluate your retrieval using NDCG@k, MAP@K ...
        ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(eval_dataset.qrels, rerank_results_model, [1, 5, 10, 100])
        embed_ndcg, embed_map, embed_recall, embed_precision = EvaluateRetrieval.evaluate(eval_dataset.qrels, rerank_results_biencoder, [1, 5, 10, 100])

        model_metrics = {
            **ndcg, **_map, **recall, **precision
        }
        biencoder_metrics = {
            **embed_ndcg, **embed_map, **embed_recall, **embed_precision
        }
        biencoder_metrics = { f"embed/{k}": v for k,v in biencoder_metrics.items() }
        return {
            **model_metrics, **biencoder_metrics
        }


    def evaluate_retrieval_datasets(self) -> Dict[str, float]:
        all_metrics = {}
        for eval_dataset_name, eval_dataset in self.retrieval_datasets.items():
            metric_key_prefix = f"eval_{eval_dataset_name}"
            metrics = self._retrieval_evaluate(
                eval_dataset=eval_dataset,
                metric_key_prefix=metric_key_prefix,
            )
                # Prefix all keys with metric_key_prefix + '_'
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
                    
            self._memory_tracker.stop_and_update_metrics(metrics)
            self.log(metrics)
            all_metrics.update(metrics)
        return all_metrics

    def _maybe_log_save_evaluate(self, *args, **kwargs):
        ####   (have to hook in here to call my special function)
        # 
        # evaluate on retrieval datasets!

        # check should_evaluate bool because it'll be reset by call to super()..
        should_evaluate = self.control.should_evaluate
        
        # do all the other stuff
        super()._maybe_log_save_evaluate(*args, **kwargs)

        # do my custom eval
        if should_evaluate:
            self.evaluate_retrieval_datasets()