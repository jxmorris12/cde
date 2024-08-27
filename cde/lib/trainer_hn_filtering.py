from typing import Tuple

from sentence_transformers import SentenceTransformer
import transformers
import torch

from cde.lib import mean_pool, print0


class NomicEmbeddingModelWrapper(torch.nn.Module):
    # TODO: Put this somewhere else
    def __init__(self):
        super().__init__()
        self.model = transformers.AutoModel.from_pretrained(
            "nomic-ai/nomic-embed-text-v1", 
            trust_remote_code=True
        )
        self.model.eval()
    
    def forward(self, **kwargs):
        output = self.model(**kwargs)
        embeddings = mean_pool(output[0], kwargs["attention_mask"])
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)


class TrainerNegativeFilterMixin:
    """Filters hard negatives based on another pre-trained model."""
            
    def _get_scores_nomic(self, query_inputs, document_inputs) -> torch.Tensor:
        from cde.lib.tensor import forward_batched
        
        if self._hn_filter_model is None:
            self._hn_filter_model = NomicEmbeddingModelWrapper()
            self._hn_filter_model.to(self.args.device)

        with torch.no_grad():
            query_outputs = forward_batched(
                model=self._hn_filter_model,
                input_ids=query_inputs["input_ids"][:, :self.model.config.max_seq_length],
                attention_mask=query_inputs["attention_mask"][:, :self.model.config.max_seq_length],
                batch_size=(self.args.per_device_train_batch_size * 4),
            )
            doc_outputs = forward_batched(
                model=self._hn_filter_model,
                input_ids=document_inputs["input_ids"][:, :512],
                attention_mask=document_inputs["attention_mask"][:, :512],
                batch_size=(self.args.per_device_train_batch_size * 4),
            )

        query_embeddings = torch.nn.functional.normalize(query_outputs, p=2, dim=1)
        doc_embeddings = torch.nn.functional.normalize(doc_outputs, p=2, dim=1)
        return query_embeddings @ doc_embeddings.T
    
    def _get_scores_stella(self, query_inputs, document_inputs) -> torch.Tensor:
        # https://huggingface.co/dunzhang/stella_en_400M_v5
        query_prompt_name = "s2p_query"
    
        queries = query_inputs["text__no_prefix"]
        docs = document_inputs["text__no_prefix"]

        if self._hn_filter_model is None:
            self._hn_filter_model = SentenceTransformer(
                "dunzhang/stella_en_400M_v5", 
                trust_remote_code=True, 
                device=self.args.device
            )
            self._hn_filter_model.max_seq_length = self.model.config.max_seq_length
            print0(f"Loaded model stella_en_400M_v5 and set max_seq_length to {self.model.config.max_seq_length}.")
        
        model = self._hn_filter_model
        with torch.no_grad():
            query_embeddings = model.encode(
                queries,
                prompt_name=query_prompt_name, 
                batch_size=(self.args.per_device_train_batch_size * 2),
                convert_to_tensor=True
            )
            doc_embeddings = model.encode(
                docs,    
                batch_size=(self.args.per_device_train_batch_size * 2),
                convert_to_tensor=True
            )
            scores = model.similarity(query_embeddings, doc_embeddings)
        return scores
    
    def _get_query_doc_scores(self, query_inputs, document_inputs) -> torch.Tensor:
        if self.args.hn_filter_model == "stella":
            return self._get_scores_stella(query_inputs, document_inputs)
        else:
            return self._get_scores_nomic(query_inputs, document_inputs)