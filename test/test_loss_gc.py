from typing import Tuple

import functools

import torch
import transformers

from cde.collate import TokenizerCollator
from cde.dataset import NomicSupervisedDataset
from cde.gradcache import GradCache
from cde.lib.misc import inputs_for_key
from cde.model import BiEncoder, DatasetTransformer


def contrastive_loss(
            model: torch.nn.Module, 
            e1: torch.Tensor, 
            e2: torch.Tensor, 
            one_hot_labels: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        e1 = e1 / e1.norm(p=2, dim=1, keepdim=True) # Query
        e2 = e2 / e2.norm(p=2, dim=1, keepdim=True) # Document
        scores = e1 @ e2.T
        print("_contrastive_loss:", scores[0:2, 0:2])

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
    tiny_config.logit_scale = 20.0
    tiny_config.disable_dropout = True
    transductive_corpus_size = 4
    tiny_config.transductive_corpus_size = transductive_corpus_size
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
        chunk_sizes=[2, 2],
        loss_fn=functools.partial(contrastive_loss, model),
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
    C = transductive_corpus_size
    query_inputs["dataset_input_ids"] = document_inputs["input_ids"][:C]
    query_inputs["dataset_attention_mask"] = document_inputs["attention_mask"][:C]
    document_inputs["dataset_input_ids"] = document_inputs["input_ids"][:C]
    document_inputs["dataset_attention_mask"] = document_inputs["attention_mask"][:C]

    document_unique_ids = document_inputs["input_ids"].sum(dim=1)
    one_hot_labels = (
        document_unique_ids[:, None] == document_unique_ids[None, :])
    
    model.eval()
    gc_loss = gc(
        query_inputs, 
        document_inputs, 
        model=model,
        model_stages=[model.first_stage_model, model.second_stage_model],
        one_hot_labels=one_hot_labels,
        no_sync_except_last=False,
        backward_fn=(lambda t: t.backward()),
        run_backward=True,
    )
    assert gc_loss

    with torch.no_grad():
        e1 = model(**query_inputs)
        e2 = model(**document_inputs)

    true_loss = contrastive_loss(
        model=model,
        e1=e1,
        e2=e2,
        one_hot_labels=one_hot_labels,
    )
    assert torch.isclose(gc_loss, true_loss)


def test_loss_gradcache__contextual():
    tiny_model_name = 'distilbert-base-uncased'
    tiny_model = transformers.AutoModel.from_pretrained(
        tiny_model_name)
    tiny_tokenizer = transformers.AutoTokenizer.from_pretrained(
        tiny_model_name
    )
    tiny_config = tiny_model.config
    tiny_config.limit_layers = None
    tiny_config.logit_scale = 20.0
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
        chunk_sizes=[2, 2],
        loss_fn=functools.partial(contrastive_loss, model),
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
    model.eval()
    gc_loss = gc(
        query_inputs, 
        document_inputs, 
        model=model,
        model_stages=[model.first_stage_model, model.second_stage_model],
        one_hot_labels=one_hot_labels,
        no_sync_except_last=False,
        backward_fn=(lambda t: t.backward()),
        run_backward=True,
    )
    assert gc_loss > 0.0
    with torch.no_grad():
        e1 = model(**query_inputs)
        e2 = model(**document_inputs)

    true_loss = contrastive_loss(
        model=model,
        e1=e1,
        e2=e2,
        one_hot_labels=one_hot_labels,
    )
    assert torch.isclose(gc_loss, true_loss)


def test_gradient_gradcache__biencoder():
    tiny_model_name = 'distilbert-base-uncased'
    tiny_model = transformers.AutoModel.from_pretrained(
        tiny_model_name)
    tiny_tokenizer = transformers.AutoTokenizer.from_pretrained(
        tiny_model_name
    )
    tiny_config = tiny_model.config
    tiny_config.limit_layers = None
    tiny_config.logit_scale = 20.0
    tiny_config.disable_dropout = True
    model = BiEncoder(
         config=tiny_config,
         embedder=tiny_model,
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
        chunk_sizes=[2, 2],
        loss_fn=functools.partial(contrastive_loss, model),
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

    gc_loss = gc(
        query_inputs, 
        document_inputs, 
        model=model,
        one_hot_labels=one_hot_labels,
        no_sync_except_last=False,
        backward_fn=(lambda t: t.backward()),
        run_backward=True,
    )
    assert gc_loss > 0.0
    gc_grad_sum = (sum([t.grad.norm(p=2) for t in model.parameters()]))
    model.zero_grad()
    true_loss = contrastive_loss(
        model=model,
        e1=model(**query_inputs),
        e2=model(**document_inputs),
        one_hot_labels=one_hot_labels,
    )
    true_loss.backward()
    assert torch.isclose(gc_loss, true_loss)

    true_grad_sum = (sum([t.grad.norm(p=2) for t in model.parameters()]))
    assert torch.isclose(true_grad_sum, gc_grad_sum)


def test_gradient_gradcache__contextual():
    tiny_model_name = 'distilbert-base-uncased'
    tiny_model = transformers.AutoModel.from_pretrained(
        tiny_model_name)
    tiny_model_2 = transformers.AutoModel.from_pretrained(
        tiny_model_name)
    tiny_tokenizer = transformers.AutoTokenizer.from_pretrained(
        tiny_model_name
    )
    tiny_config = tiny_model.config
    tiny_config.limit_layers = None
    tiny_config.logit_scale = 20.0
    tiny_config.disable_dropout = True
    model = DatasetTransformer(
         config=tiny_config,
         embedder=tiny_model,
         dataset_backbone=tiny_model_2,
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
        chunk_sizes=[2, 2],
        loss_fn=functools.partial(contrastive_loss, model),
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
    model.eval()

    gc_loss = gc(
        query_inputs, 
        document_inputs, 
        model=model,
        model_stages=[model.first_stage_model, model.second_stage_model],
        one_hot_labels=one_hot_labels,
        no_sync_except_last=False,
        backward_fn=(lambda t: t.backward()),
        run_backward=True,
    )
    assert gc_loss > 0.0
    gc_grad_sums = { name: t.grad.norm(p=2) for name, t in model.named_parameters() }
    model.zero_grad()

    e1 = model(**query_inputs)
    e2 = model(**document_inputs)

    true_loss = contrastive_loss(
        model=model,
        e1=e1,
        e2=e2,
        one_hot_labels=one_hot_labels,
    )
    true_loss.backward()
    assert torch.isclose(gc_loss, true_loss)

    true_grad_sums = { name: t.grad.norm(p=2) for name, t in model.named_parameters() }

    sums_diff = { 
        param_name: (gc_grad_sums[param_name] - true_grad_sums[param_name]).item() 
        for param_name in true_grad_sums.keys() 
    }
    for param_name in gc_grad_sums.keys():
        assert torch.isclose(
            gc_grad_sums[param_name], 
            true_grad_sums[param_name],
        )
