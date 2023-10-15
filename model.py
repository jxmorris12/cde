import torch
import transformers


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
    backbone: transformers.PreTrainedModel
    def __init__(
            self, 
            config, #: transformers.PreTrainedConfig, 
            embedder: transformers.PreTrainedModel, 
            backbone: transformers.PreTrainedModel
        ):
        super().__init__(config=config)
        self.embedder = embedder

        # TODO make this a little nicer. (Not every model has 'embeddings...')
        self.backbone = backbone
        self.backbone.embeddings.position_embeddings.weight.requires_grad = False
        self.backbone.embeddings.position_embeddings.weight.fill_(0.0)

        # TODO - fix. consider BART or another encoder-decoder.
        # self.backbone = transformers.AutoModel.from_pretrained("t5-small")
        # for block in self.backbone.decoder.block: 
        #     block.layer[0].SelfAttention.has_relative_attention_bias = False

        embedding_dim = 768
        self.hidden_size = self.backbone.config.hidden_size

        self.n_sequence = 16
        self.query_projection = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, self.hidden_size * self.n_sequence)
        )
        self.corpus_projection = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, self.hidden_size)
        )

        # whether to share hard negatives between queries.
        # TODO argparse for this.
        self.share_hard_negatives = False
    
    def forward_embedder(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            embedding: torch.Tensor = None,
    ) -> torch.Tensor:
        outputs = (
            self.embedder(
                input_ids=input_ids,
                attention_mask=attention_mask).last_hidden_state
        )
        return mean_pool(outputs, attention_mask)


    def forward(self, query_embedding: torch.Tensor, document_embeddings: torch.Tensor) -> torch.Tensor:
        batch_size = query_embedding.shape[0]
        input_query_embedding = query_embedding # save for biencoder
        query_embedding = self.query_projection(query_embedding)
        query_embedding = query_embedding.reshape((batch_size, self.n_sequence, self.hidden_size))
        assert query_embedding.shape == (batch_size, self.n_sequence, self.hidden_size)

        document_embeddings = self.corpus_projection(document_embeddings)
        if self.share_hard_negatives:
            document_embeddings = document_embeddings.repeat((batch_size, 1, 1))
        else:
            inbatch_document_embeddings = document_embeddings[:batch_size, :]
            inbatch_document_embeddings = (
                inbatch_document_embeddings[:, None, :]
                    .repeat((1, batch_size, 1))
            )
            hn_document_embeddings = (
                document_embeddings[batch_size:, :].reshape((batch_size, -1, self.hidden_size))
            )
            document_embeddings = torch.cat((
                inbatch_document_embeddings, hn_document_embeddings
            ), dim=1)
        
        _, corpus_size, hidden_dim = document_embeddings.shape
        assert _ == batch_size

        if self.config.architecture == "query_dependent":
            inputs_embeds = torch.cat((query_embedding, document_embeddings), dim=1)
            output = self.backbone(
                inputs_embeds=inputs_embeds,
                attention_mask=torch.ones((batch_size, self.n_sequence + corpus_size), dtype=torch.long, device=query_embedding.device),
            )
            output_vectors = output.last_hidden_state[:, self.n_sequence:, :]
            query_output_vectors = output.last_hidden_state[:, :self.n_sequence, :].mean(dim=1)
            scores = torch.bmm(output_vectors, query_output_vectors[:,:, None])
            scores = scores.squeeze(dim=2)
        elif self.config.architecture == "query_independent":
            output = self.backbone(
                inputs_embeds=document_embeddings,
                attention_mask=torch.ones((batch_size, corpus_size), dtype=torch.long, device=query_embedding.device),
            )
            output_vectors = output.last_hidden_state
            query_embedding = query_embedding.mean(dim=1) # Mean along sequence length (it's really just a projection then.)
            assert query_embedding.shape == (batch_size, hidden_dim)
            scores = torch.einsum('bd,bcd->bc', query_embedding, output_vectors)
        elif self.config.architecture == "biencoder_extended":
            inputs_embeds = document_embeddings
            inputs_embeds = inputs_embeds.reshape((batch_size * corpus_size, 1, hidden_dim))
            output = self.backbone(
                inputs_embeds=inputs_embeds,
                attention_mask=torch.ones((batch_size * corpus_size, 1), dtype=torch.long, device=query_embedding.device),
            )
            output_vectors = output.last_hidden_state
            output_vectors = output_vectors.squeeze(1).reshape((batch_size, corpus_size, hidden_dim))

            query_embedding = query_embedding.mean(dim=1) # Mean along sequence length (it's really just a projection then.)
            assert query_embedding.shape == (batch_size, hidden_dim)
            
            scores = torch.einsum('bd,bcd->bc', query_embedding, output_vectors)
        elif self.config.architecture == "biencoder":
            scores = torch.einsum('bd,bcd->bc', input_query_embedding, document_embeddings)
        else:
            raise ValueError(f"unknown architecture {self.config.architecture}")

        return scores
