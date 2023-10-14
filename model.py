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
    def __init__(self, config, embedder: transformers.PreTrainedModel):
        super().__init__(config=config)
        self.embedder = embedder

        # TODO: argparse for this --v
        # self.transformer = transformers.AutoModel.from_pretrained("distilbert-base-uncased")
        self.transformer = transformers.AutoModel.from_pretrained("bert-base-uncased")
        self.transformer.embeddings.position_embeddings.weight.requires_grad = False # don't want position embeddings
        self.transformer.embeddings.position_embeddings.weight.fill_(0.0)

        # TODO - fix. consider BART?
        # self.transformer = transformers.AutoModel.from_pretrained("t5-small")
        # for block in self.transformer.decoder.block: 
        #     block.layer[0].SelfAttention.has_relative_attention_bias = False

        embedding_dim = 768
        self.hidden_size = self.transformer.config.hidden_size

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
        batch_size, corpus_size, hidden_dim = document_embeddings.shape
        query_embedding = self.query_projection(query_embedding)
        query_embedding = query_embedding.reshape((batch_size, self.n_sequence, self.hidden_size))
        assert query_embedding.shape == (batch_size, self.n_sequence, self.hidden_size)

        document_embeddings = self.corpus_projection(document_embeddings)
        assert document_embeddings.shape == (batch_size, corpus_size, self.hidden_size)

        if self.config.architecture == "query_dependent":
            inputs_embeds = torch.cat((query_embedding, document_embeddings), dim=1)
            output = self.transformer(
                inputs_embeds=inputs_embeds,
                attention_mask=torch.ones((batch_size, self.n_sequence + corpus_size), dtype=torch.long, device=query_embedding.device),
            )
            output_vectors = output.last_hidden_state[:, self.n_sequence:, :]
            query_output_vectors = output.last_hidden_state[:, :self.n_sequence, :].mean(dim=1)
            scores = torch.bmm(output_vectors, query_output_vectors[:,:, None])
            scores = scores.squeeze(dim=2)
        elif self.config.architecture == "query_independent":
            output = self.transformer(
                inputs_embeds=document_embeddings,
                attention_mask=torch.ones((batch_size, corpus_size), dtype=torch.long, device=query_embedding.device),
            )
            output_vectors = output.last_hidden_state
            query_embedding = query_embedding.mean(dim=1) # Mean along sequence length (it's really just a projection then.)
            assert query_embedding.shape == (batch_size, hidden_dim)
            scores = torch.bmm(output_vectors, query_embedding[:,:, None])
            scores = scores.squeeze(dim=2) # TODO: verify via test.
        elif self.config.architecture == "biencoder_extended":
            inputs_embeds = document_embeddings
            inputs_embeds = inputs_embeds.reshape((batch_size * corpus_size, 1, hidden_dim))
            output = self.transformer(
                inputs_embeds=inputs_embeds,
                attention_mask=torch.ones((batch_size * corpus_size, 1), dtype=torch.long, device=query_embedding.device),
            )
            output_vectors = output.last_hidden_state
            output_vectors = output_vectors.squeeze(1).reshape((batch_size, corpus_size, hidden_dim))

            query_embedding = query_embedding.mean(dim=1) # Mean along sequence length (it's really just a projection then.)
            assert query_embedding.shape == (batch_size, hidden_dim)
            
            scores = torch.bmm(output_vectors, query_embedding[:,:, None])
            scores = scores.squeeze(dim=2)
        elif self.config.architecture == "biencoder":
            scores = query_embedding @ document_embeddings.T # TODO verify
        else:
            raise ValueError(f"unknown architecture {self.config.architecture}")

        # output = self.transformer(
        #     inputs_embeds=query_embedding,
        #     attention_mask=torch.ones((batch_size, self.n_sequence), dtype=torch.long, device=query_embedding.device),
        #     decoder_inputs_embeds=document_embeddings,
        #     decoder_attention_mask=torch.ones((batch_size, corpus_size), dtype=torch.long, device=query_embedding.device),
        # )
        # output_vectors = output.last_hidden_state

        return scores

        # scores = self.score(output_vectors)
        # assert scores.shape == (batch_size, corpus_size, 1)
        # return scores.squeeze(2)

 
# We construct the SentenceTransformer bi-encoder from scratch with mean-pooling
# model_name = "distilbert-base-uncased" 
# word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
# pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
# model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# #### Provide a high batch-size to train better with triplets!
# retriever = TrainRetriever(model=model, batch_size=train_batch_size)
