import sys
sys.path.append('/home/paperspace/tti3')

from typing import Tuple

import functools

import torch
import transformers

from collate import TokenizerCollator
from dataset import NomicSupervisedDataset
from gradcache import GradCache
from lib.misc import inputs_for_key
from model import DatasetTransformer


def contrastive_loss(
            model: torch.nn.Module, 
            e1: torch.Tensor, 
            e2: torch.Tensor, 
            one_hot_labels: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        e1 = e1 / e1.norm(p=2, dim=1, keepdim=True) # Query
        e2 = e2 / e2.norm(p=2, dim=1, keepdim=True) # Document
        scores = e1 @ e2.T

        batch_size, _ = scores.shape
        # This multiply-by-20 thing is used in BeIR and LaPRador:
        # "When calculating the loss, we apply a re-scaling trick
        # of multiplying the cosine similarity score by 20 for
        # better optimization (Thakur et al., 2021)."
        # 
        scores *= model.temp

        assert scores.shape == one_hot_labels.shape
        labels = one_hot_labels / one_hot_labels.sum(dim=1, keepdim=True)

        loss = torch.nn.functional.cross_entropy(
            scores, labels, label_smoothing=0.0
        )
        if (loss.isnan()):
            raise RuntimeError("Loss is nan!")
        
        pred_labels = one_hot_labels[torch.arange(len(one_hot_labels)), scores.argmax(dim=1)]
        acc = pred_labels.float().mean()

        metrics = {
            "acc": acc.item(),
            "stats_total_queries": len(e1),
            "stats_total_documents": len(e2),
            "batch_size": batch_size,
        }
        return loss

def test_loss_gradcache():
    tiny_model_name = 'distilbert-base-uncased'
    tiny_model = transformers.AutoModel.from_pretrained(
        tiny_model_name)
    tiny_tokenizer = transformers.AutoTokenizer.from_pretrained(
        tiny_model_name
    )
    tiny_config = tiny_model.config
    tiny_config.limit_layers = None
    tiny_config.contrastive_temp = 20.0
    tiny_config.disable_dropout = True
    model = DatasetTransformer(
         config=tiny_config,
         embedder=tiny_model,
         dataset_backbone=tiny_model,
    )
    dataset = NomicSupervisedDataset(
        tokenizer=tiny_tokenizer,
        num_hard_negatives=0,
        max_seq_length=16,
    )
    data_collator = TokenizerCollator(
        tokenizer=tiny_tokenizer,
        padding='longest',
        return_tensors='pt',
        max_length=16,
    )
    gc = GradCache(
        model=model,
        chunk_sizes=2,
        loss_fn=functools.partial(contrastive_loss, model),
        fp16=False,
        scaler=None,
        backward_fn=(lambda t: t.backward()),
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=8,
        collate_fn=data_collator,
        num_workers=0,
        pin_memory=True,
    )
    dataloader_iter = iter(dataloader)
    inputs = next(dataloader_iter)
    query_inputs = inputs_for_key(inputs, key="query")
    document_inputs = inputs_for_key(inputs, key="document")
    ##########################################################################################
    query_inputs["dataset_input_ids"] = document_inputs["input_ids"]
    query_inputs["dataset_attention_mask"] = document_inputs["attention_mask"]
    document_inputs["dataset_input_ids"] = document_inputs["input_ids"]
    document_inputs["dataset_attention_mask"] = document_inputs["attention_mask"]

    document_unique_ids = document_inputs["input_ids"].sum(dim=1)
    one_hot_labels = (
        document_unique_ids[:, None] == document_unique_ids[None, :])

    loss = gc(
        query_inputs, 
        document_inputs, 
        one_hot_labels=one_hot_labels,
        no_sync_except_last=False
    )
    assert loss > 0.0