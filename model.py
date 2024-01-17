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
            dataset_backbone: transformers.PreTrainedModel,
        ):
        super().__init__(config=config)

        if config.limit_layers:
            print(f"Limiting layers to {config.limit_layers}")
            embedder.transformer.layer = embedder.transformer.layer[:config.limit_layers]
            dataset_embedder.transformer.layer = dataset_embedder.transformer.layer[:config.limit_layers]
            dataset_backbone.transformer.layer = dataset_backbone.transformer.layer[:config.limit_layers]
        self.embedder = embedder

        # TODO make this a little nicer. (Not every model has 'embeddings...')
        self.dataset_backbone = dataset_backbone
        self.dataset_backbone.embeddings.position_embeddings.weight.requires_grad = False
        self.dataset_backbone.embeddings.position_embeddings.weight.fill_(0.0)

        # TODO - fix. consider BART or another encoder-decoder.
        # self.backbone = transformers.AutoModel.from_pretrained("t5-small")
        # for block in self.backbone.decoder.block: 
        #     block.layer[0].SelfAttention.has_relative_attention_bias = False

        self.hidden_size = self.embedder.config.hidden_size
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size * 2, self.hidden_size * 4),
            torch.nn.GELU(),
            torch.nn.Linear(self.hidden_size * 4, self.hidden_size * 2),
            torch.nn.GELU(),
            torch.nn.Linear(self.hidden_size * 2, self.hidden_size),
        )

        self.gamma = config.gamma
        if config.disable_dropout:
            disable_dropout(self)
    
    def forward_embedder(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            dataset_input_ids: torch.Tensor,
            dataset_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        dataset_input_embeddings = mean_pool(
            self.dataset_embedder(
                input_ids=dataset_input_ids,
                attention_mask=dataset_attention_mask
            )
        )
        assert len(dataset_input_embeddings.shape) == 2 # (b, d)
        dataset_input_embeddings = dataset_input_embeddings[:, None, :]
        dataset_embedding = mean_pool(
            self.dataset_backbone(
                input_ids=dataset_input_embeddings
            )
        )
        batch_size = dataset_input_ids.shape[0]
        embeddings = (
            self.embedder(
                input_ids=input_ids,
                attention_mask=attention_mask).last_hidden_state
        )
        mlp_input_embeddings = torch.cat(
            (
                dataset_embedding.repeat((batch_size, 1)),
                embeddings
            )
        )
        # TODO use self.gamma here
        # output_vectors = (document_embeddings * self.gamma) + (output_vectors * (1 - self.gamma))
        return self.mlp(mlp_input_embeddings)


    def forward(self, query_embedding: torch.Tensor, document_embeddings: torch.Tensor) -> torch.Tensor:
        batch_size = query_embedding.shape[0]
        query_embedding = self.query_projection(query_embedding)
        query_embedding = query_embedding.reshape((batch_size, self.n_sequence, self.hidden_size))
        assert query_embedding.shape == (batch_size, self.n_sequence, self.hidden_size)
        
        pos_document_embeddings = document_embeddings[:batch_size, :]
        pos_document_embeddings = (
            pos_document_embeddings[None, :, :]
                .repeat((batch_size, 1, 1))
        )
        if len(document_embeddings) < batch_size:
            # handle hard negatives
            hn_document_embeddings = (
                document_embeddings[batch_size:, :]
                )
            hn_document_embeddings = (
                hn_document_embeddings.reshape((batch_size, -1, self.hidden_size))
            )
            document_embeddings = torch.cat((
                pos_document_embeddings, hn_document_embeddings
            ), dim=1)
        _, corpus_size, hidden_dim = document_embeddings.shape
        inputs_embeds = document_embeddings
        inputs_embeds = inputs_embeds.reshape((batch_size * corpus_size, 1, hidden_dim))
        output = self.backbone(
            inputs_embeds=inputs_embeds,
        )
        output_vectors = output.last_hidden_state[:, self.n_sequence:, :]
        output_vectors = output_vectors.reshape((batch_size, corpus_size, hidden_dim))
        
        query_embedding = query_embedding[:, 0, :]
        scores = torch.einsum('bd,bcd->bc', query_embedding, output_vectors)

        return scores
        