import torch
import transformers

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # TODO argparse stuff
        self.transformer = transformers.AutoModel.from_pretrained("t5-small")
        for layer in self.transformer.layers: 
            self.has_relative_attention_bias = False


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
        self.score = torch.nn.Linear((self.hidden_size, 1))

    def forward(self, query_embeddings: torch.Tensor, corpus_embeddings: torch.Tensor) -> torch.Tensor:
        batch_size, corpus_size, hidden_dim = corpus_embeddings.shape
        query_embeddings = self.query_projection(query_embeddings)
        query_embeddings = query_embeddings.reshape((-1, self.n_sequence, self.hidden_size))

        corpus_embeddings = self.corpus_projection(corpus_embeddings)


        output = self.transformer(
            inputs_embeds=query_embeddings,
            attention_mask=torch.ones((batch_size, self.n_sequence), dtype=torch.long, device=query_embeddings.device),
            decoder_inputs_embeds=corpus_embeddings,
            decoder_attention_mask=torch.ones((batch_size, corpus_size), dtype=torch.long, device=query_embeddings.device)
        )

        return self.score(output.last_hidden_state)

 
# We construct the SentenceTransformer bi-encoder from scratch with mean-pooling
# model_name = "distilbert-base-uncased" 
# word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
# pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
# model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# #### Provide a high batch-size to train better with triplets!
# retriever = TrainRetriever(model=model, batch_size=train_batch_size)
