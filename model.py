import torch
import transformers


def disable_dropout(model: torch.nn.Module):
    dropout_modules = [m for m in model.modules() if isinstance(m, torch.nn.Dropout)]
    for m in dropout_modules:
        m.p = 0.0
    print(
        f"Disabled {len(dropout_modules)} dropout modules from model type {type(model)}"
    )


def mean_pool(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    B, S, D = hidden_states.shape
    unmasked_outputs = hidden_states * attention_mask[..., None]
    pooled_outputs = unmasked_outputs.sum(dim=1) / attention_mask.sum(dim=1)[:, None]
    assert pooled_outputs.shape == (B, D)
    return pooled_outputs


class Model(transformers.PreTrainedModel):
    embedder: transformers.PreTrainedModel
    dataset_embedder: transformers.PreTrainedModel
    dataset_backbone: transformers.PreTrainedModel
    def __init__(
            self, 
            config, #: transformers.PreTrainedConfig, 
            embedder: transformers.PreTrainedModel, 
            dataset_embedder: transformers.PreTrainedModel, 
        ):
        super().__init__(config=config)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased')

        if config.limit_layers:
            print(f"Limiting layers to {config.limit_layers}")
            embedder.transformer.layer = embedder.transformer.layer[:config.limit_layers]
            dataset_embedder.transformer.layer = dataset_embedder.transformer.layer[:config.limit_layers]
        
        # TODO: fix arguments. Disable t5 positional embedder.
        self.embedder = transformers.AutoModel.from_pretrained('t5-base')
        self.dataset_embedder = dataset_embedder

        # TODO - fix. consider BART or another encoder-decoder.
        # self.backbone = transformers.AutoModel.from_pretrained("t5-small")
        disabled_attention_bias_count = 0
        for M in self.embedder.encoder.modules(): 
            if hasattr(M, "has_relative_attention_bias"):
                setattr(M, "has_relative_attention_bias", False)
                # print("> disabled encoder bias")
                disabled_attention_bias_count += 1
        for M in self.embedder.decoder.modules():
            if isinstance(M, transformers.models.t5.modeling_t5.T5LayerCrossAttention):
                setattr(M, "has_relative_attention_bias", False)
                # print("> disabled decoder bias")
                disabled_attention_bias_count += 1
        print(f"Disabled {disabled_attention_bias_count} attention biases")

        self.hidden_size = self.embedder.config.hidden_size
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, self.hidden_size * 2),
            torch.nn.GELU(),
            torch.nn.Linear(self.hidden_size * 2, self.hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
        )
        if config.disable_dropout:
            disable_dropout(self)
        
        self.dataset_positional_embeddings = torch.nn.Parameter(
            torch.randn((1024, self.hidden_size)), requires_grad=True
        )
    
    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            dataset_input_ids: torch.Tensor,
            dataset_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # print("shapes:", input_ids.shape, dataset_input_ids.shape, "(", attention_mask.shape, dataset_attention_mask.shape, ")")
        batch_size = dataset_input_ids.shape[0]
        dataset_outputs = self.dataset_embedder(
            input_ids=dataset_input_ids,
            attention_mask=dataset_attention_mask
        ).last_hidden_state
        dataset_embeddings = mean_pool(
            hidden_states=dataset_outputs,
            attention_mask=dataset_attention_mask,
        )
        assert len(dataset_embeddings.shape) == 2 # (b, d)
        dataset_embeddings = (
            dataset_embeddings + self.dataset_positional_embeddings[:batch_size, :]
        )
        dataset_embeddings = dataset_embeddings[None, :, :].expand(batch_size, -1, -1)
        batch_size = dataset_input_ids.shape[0]
        outputs = self.embedder(
            inputs_embeds=dataset_embeddings,
            decoder_input_ids=input_ids,
            decoder_attention_mask=attention_mask,
        )
        # select last hidden token
        gather_idxs = attention_mask.cumsum(1).argmax(1)
        batch_idxs = torch.arange(batch_size, device=gather_idxs.device)
        embeddings = outputs.last_hidden_state[batch_idxs, gather_idxs]
        assert embeddings.shape == (batch_size, self.hidden_size)
        # project
        output_embeddings = self.mlp(embeddings)
        assert output_embeddings.shape == (batch_size, self.hidden_size)
        return output_embeddings
