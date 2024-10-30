import pytest
import torch
import transformers

from cde.lib import load_embedder_and_tokenizer, ContextualModelConfig
from cde.model import DatasetTransformer
from cde.run_args import ModelArguments


def test_dataset_transformer_invariance_synthetic():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA found")
    model_config = ContextualModelConfig(**vars(ModelArguments()))
    model, _ = load_embedder_and_tokenizer("nomic-ai/nomic-bert-2048")
    backbone, _ = load_embedder_and_tokenizer("nomic-ai/nomic-bert-2048")
    model_config.transductive_corpus_size = 2
    model_config.limit_layers = 1
    dt = DatasetTransformer(
        config=model_config,
        embedder=model,
        dataset_backbone=backbone,
    ).cuda()
    dt.eval()

    i1 = torch.tensor([
        [1, 2, 3, 4],
        [5, 6, 7, 8]], dtype=torch.long, device="cuda")
    a1 = torch.ones_like(i1)
    i1_reordered = i1[[1, 0], :]
    print("SHAPES", i1.shape, a1.shape, "/", i1_reordered.shape)

    i2 = torch.tensor([
        [5, 19, 13, 17, 91, 11],
        [17, 91, 11, 5, 19, 13]
    ], dtype=torch.long, device="cuda")
    a2 = torch.ones_like(i2, device="cuda")

    with torch.autocast("cuda", dtype=torch.bfloat16):
        print("INPUT SHAPES:", i2.shape, a2.shape, "/", i1_reordered.shape, a1.shape)
        print("O1")
        o1 = dt(
            input_ids=i2,
            attention_mask=a2,
            dataset_input_ids=i1_reordered,
            # dataset_input_ids=i1,
            dataset_attention_mask=a1,
        )
        print("O2")
        o2 = dt(
            input_ids=i2,
            attention_mask=a2,
            dataset_input_ids=i1_reordered,
            dataset_attention_mask=a1,
        )
        # o3 = dt(
        #     input_ids=i2,
        #     attention_mask=a2,
        #     dataset_input_ids=i1_reordered,
        #     dataset_attention_mask=a1,
        # )
    torch.testing.assert_close(o1, o2)


def test_dataset_transformer_invariance_real():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA found")
    model_config = ContextualModelConfig(**vars(ModelArguments()))
    mn = "nomic-ai/nomic-bert-2048"
    model, _ = load_embedder_and_tokenizer(mn)
    backbone, _ = load_embedder_and_tokenizer(mn)
    tokenizer = transformers.AutoTokenizer.from_pretrained(mn)
    model_config.transductive_corpus_size = 4
    dt = DatasetTransformer(
        config=model_config,
        embedder=model,
        dataset_backbone=backbone,
    ).cuda()
    dt.eval()

    t1 = tokenizer(
        [
            "One dataset example input",
            "Seven Tyrannosaurus Rex Dinosaurs"
        ],
        return_tensors="pt",
        padding=True,
    ).to("cuda")
    i1 = t1.input_ids
    a1 = t1.attention_mask

    i1_reordered = i1[[1, 0], :]
    a1_reordered = a1[[1, 0], :]
    
    t2 = tokenizer(
        [
            "It's hard to write example sentences",
            "so sometimes u just have to get meta with it i guess"
        ],
        return_tensors="pt",
        padding=True,
    ).to("cuda")
    i2 = t2.input_ids
    a2 = t2.attention_mask

    with torch.autocast("cuda", dtype=torch.bfloat16):
        print("O1")
        o1 = dt(
            input_ids=i2,
            attention_mask=a2,
            dataset_input_ids=i1_reordered,
            dataset_attention_mask=a1_reordered,
        )
        print("O2")
        o2 = dt(
            input_ids=i2,
            attention_mask=a2,
            dataset_input_ids=i1,
            dataset_attention_mask=a1,
        )
    torch.testing.assert_close(o1, o2)

