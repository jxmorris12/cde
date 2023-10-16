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
        self.prompt_projection = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, self.hidden_size * self.n_sequence)
        )
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
        self.share_hard_negatives = True
    
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
        """
        query_embedding (float torch.Tensor) - shape (batch_size, embedding_dim)
        document_embeddings (float torch.Tensor) - shape (corpus_size, embedding_dim)
            where the corpus_size >= batch_size and is structured like this:
                [d1, d2, d3, hn1_1, hn1_2, hn2_1, hn2_2, hn3_1, hn3_2]
                for a corpus with three documents and two hard negatives per document
        """
        
        batch_size = query_embedding.shape[0]
        input_query_embedding = query_embedding # save for biencoder

        query_embedding = self.query_projection(query_embedding)
        query_embedding = query_embedding.reshape((batch_size, self.n_sequence, self.hidden_size))
        assert query_embedding.shape == (batch_size, self.n_sequence, self.hidden_size)

        if self.share_hard_negatives:
            document_embeddings = document_embeddings[None].repeat((batch_size, 1, 1))
        else:
            pos_document_embeddings = document_embeddings[:batch_size, :]
            pos_document_embeddings = (
                pos_document_embeddings[None, :, :]
                    .repeat((batch_size, 1, 1))
            )
            hn_document_embeddings = (
                document_embeddings[batch_size:, :]
                )
            hn_document_embeddings = (
                hn_document_embeddings.reshape((batch_size, -1, self.hidden_size))
            )
            document_embeddings = torch.cat((
                pos_document_embeddings, hn_document_embeddings
            ), dim=1)
        
        input_document_embeddings = document_embeddings # save for biencoder
        document_embeddings = self.corpus_projection(document_embeddings)
        _, corpus_size, hidden_dim = document_embeddings.shape
        assert _ == batch_size

        soft_prompt = torch.ones((1, self.hidden_size), device=query_embedding.device, dtype=torch.float32)
        soft_prompt = self.prompt_projection(soft_prompt).reshape((1, self.n_sequence, self.hidden_size))
        if self.config.architecture == "query_dependent":
            soft_prompt = soft_prompt.repeat((batch_size, 1, 1))
            inputs_embeds = torch.cat((soft_prompt, query_embedding, document_embeddings), dim=1)
            output = self.backbone(
                inputs_embeds=inputs_embeds,
                attention_mask=torch.ones((batch_size, (2 * self.n_sequence) + corpus_size), dtype=torch.long, device=query_embedding.device),
            )
            output_vectors = output.last_hidden_state[:, (2*self.n_sequence):, :]
            query_output_vectors = output.last_hidden_state[:, (self.n_sequence):(2*self.n_sequence), :].mean(dim=1)
            scores = torch.bmm(output_vectors, query_output_vectors[:, :, None])
            scores = scores.squeeze(dim=2)
        elif self.config.architecture == "query_independent":
            soft_prompt = soft_prompt.repeat((batch_size, 1, 1))
            
            inputs_embeds = document_embeddings
            inputs_embeds = torch.cat((soft_prompt, inputs_embeds), dim=1)
            output = self.backbone(
                inputs_embeds=inputs_embeds,
                attention_mask=torch.ones((batch_size, self.n_sequence + corpus_size), dtype=torch.long, device=query_embedding.device),
            )
            output_vectors = output.last_hidden_state[:, self.n_sequence:, :]
            scores = torch.einsum('bd,bcd->bc', input_query_embedding, output_vectors)
        elif self.config.architecture == "biencoder_extended":
            soft_prompt = soft_prompt.repeat((batch_size * corpus_size, 1, 1))
            inputs_embeds = document_embeddings
            inputs_embeds = inputs_embeds.reshape((batch_size * corpus_size, 1, hidden_dim))
            inputs_embeds = torch.cat((soft_prompt, inputs_embeds), dim=1)
            output = self.backbone(
                inputs_embeds=inputs_embeds,
                attention_mask=torch.ones((batch_size * corpus_size, 1), dtype=torch.long, device=query_embedding.device),
            )
            output_vectors = output.last_hidden_state[:, self.n_sequence:, :]
            output_vectors = output_vectors.reshape((batch_size, corpus_size, hidden_dim))
            scores = torch.einsum('bd,bcd->bc', input_query_embedding, output_vectors)
        elif self.config.architecture == "biencoder":
            # here we just throw away the projections
            # input_query_embedding /= input_query_embedding.norm(p=2, dim=-1, keepdim=True)
            # document_embeddings /= input_document_embeddings.norm(p=2, dim=-1, keepdim=True)
            scores = torch.einsum('bd,bcd->bc', input_query_embedding, input_document_embeddings)
        else:
            raise ValueError(f"unknown architecture {self.config.architecture}")


        assert scores.shape == (batch_size, corpus_size), f"got invalid scores of shape {scores.shape}"
        return scores
