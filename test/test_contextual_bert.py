import torch
import transformers


from cde.lib import ContextualModelConfig, load_embedder_and_tokenizer
from cde.model import ContextualCrossAttention

from cde.lib.contextual_bert import ContextualBertModel
from cde.lib.contextual_bert.configuration import ContextualBertConfig

device = ("cuda" if torch.cuda.is_available() else "cpu")

def test_contextual_bert():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA found")
    config = ContextualBertConfig()
    model = ContextualBertModel(config=config).to(device)

    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")

    tokens = tokenizer(["Abby is cute", "do you agree?"], padding=True, truncation=True, return_tensors="pt").to(device)

    batch_size = tokens["input_ids"].shape[0]
    encoder_hidden_states = torch.zeros(
        (batch_size, 32, config.hidden_size), device=device, dtype=torch.float32
    )
    with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.bfloat16):
        output = model(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
            encoder_hidden_states=encoder_hidden_states,
        )
    assert output.pooler_output.shape == (2, 768)
    assert output.last_hidden_state.shape == (2, 6, 768)


def test_contextual_cross_attention():
    embedder, tokenizer = load_embedder_and_tokenizer("distilbert-base-uncased")
    model_config = ContextualModelConfig(
        limit_layers=False,
        disable_dropout=False,
        logit_scale=50,
        transductive_tie_token_embeddings=False,
        transductive_corpus_size=2, 
        tokens_per_document=3,
    )
    model = ContextualCrossAttention(
        config=model_config,
        embedder=embedder
    ).to(device)
    tokens = tokenizer(["Abby is cute", "do you agree?"], padding=True, truncation=True, return_tensors="pt").to(device)
    tokens_dataset = tokenizer(["This code is complicated", "but what did we expect?"], padding=True, truncation=True, return_tensors="pt").to(device)

    with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.bfloat16):
        output = model(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
            dataset_input_ids=tokens_dataset["input_ids"],
            dataset_attention_mask=tokens_dataset["attention_mask"],
            output_hidden_states=True,
        )
    
    assert output["hidden_states"].shape == (2, 6, 768)
        