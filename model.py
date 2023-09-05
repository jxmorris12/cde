import torch
import transformers

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # TODO argparse stuff
        self.transformer = transformers.AutoModel.from_pretrained("t5-small")
        for block in self.transformer.decoder.block: 
            block.layer[0].SelfAttention.has_relative_attention_bias = False

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
        self.score = torch.nn.Linear(self.hidden_size, 1)

    def forward(self, query_embedding: torch.Tensor, document_embeddings: torch.Tensor) -> torch.Tensor:
        batch_size, corpus_size, hidden_dim = document_embeddings.shape
        query_embedding = self.query_projection(query_embedding)
        query_embedding = query_embedding.reshape((batch_size, self.n_sequence, self.hidden_size))
        assert query_embedding.shape == (batch_size, self.n_sequence, self.hidden_size)

        document_embeddings = self.corpus_projection(document_embeddings)
        assert document_embeddings.shape == (batch_size, corpus_size, self.hidden_size)

        output = self.transformer(
            inputs_embeds=query_embedding,
            attention_mask=torch.ones((batch_size, self.n_sequence), dtype=torch.long, device=query_embedding.device),
            decoder_inputs_embeds=document_embeddings,
            decoder_attention_mask=torch.ones((batch_size, corpus_size), dtype=torch.long, device=query_embedding.device)
        )

        scores = self.score(output.last_hidden_state)
        assert scores.shape == (batch_size, corpus_size, 1)
        return scores.squeeze(2)

 
# We construct the SentenceTransformer bi-encoder from scratch with mean-pooling
# model_name = "distilbert-base-uncased" 
# word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
# pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
# model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# #### Provide a high batch-size to train better with triplets!
# retriever = TrainRetriever(model=model, batch_size=train_batch_size)
