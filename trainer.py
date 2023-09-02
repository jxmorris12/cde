from typing import Dict, Tuple, Union
import torch
import transformers

from models import Model

class CustomTrainer(transformers.Trainer):
    def compute_loss(
        self,
        model: Model,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
    ) -> Union[Tuple[torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor]:
        batch_size = inputs["query_embeddings"].shape[0]
        scores = self.model(
            query_embeddings=inputs["query_embeddings"],
            corpus_embeddings=inputs["corpus_embeddings"][None],
        )
        labels = torch.arange((batch_size,), dtype=torch.long, device=scores.device)
        return torch.nn.functional.cross_entropy(scores, labels)