import torch
import transformers


from spider.lib.contextual_bert import ContextualBertModel
from spider.lib.contextual_bert.configuration import ContextualBertConfig

device = ("cuda" if torch.cuda.is_available() else "cpu")

def test_contextual_bert():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA found")
    config = ContextualBertConfig()
    model = ContextualBertModel(config=config).to(device)

    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")

    tokens = tokenizer("Abby is cute", return_tensors="pt").to(device)

    encoder_hidden_states = torch.zeros(
        (1, 32, config.hidden_size), device=device, dtype=torch.float32
    )
    with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.bfloat16):
        output = model(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
            encoder_hidden_states=encoder_hidden_states,
        )
    breakpoint()