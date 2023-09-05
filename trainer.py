from typing import Dict, Tuple, Union
import torch
import transformers

from model import Model

class CustomTrainer(transformers.Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._signature_columns = [
            "query_embedding", "document_embeddings", "negative_document_embeddings"]

    def compute_loss(
        self,
        model: Model,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
    ) -> Union[Tuple[torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor]:

        batch_size = inputs["query_embedding"].shape[0]

        all_document_embeddings = torch.cat(
            (inputs["document_embeddings"], inputs["negative_document_embeddings"]), dim=0
        )
        scores = self.model(
            query_embedding=inputs["query_embedding"],
            document_embeddings=all_document_embeddings[None].repeat((batch_size, 1, 1)),
        )
        labels = torch.arange(batch_size, dtype=torch.long, device=scores.device)

        original_acc = ((
            torch.nn.functional.cosine_similarity(inputs["query_embedding"][:,None], all_document_embeddings[None], dim=2)
        ).argmax(1) == labels).float().mean()
        new_acc = (scores.argmax(1) == labels).float().mean()

        # print(f"acc: {original_acc.item()*100:.1f} / now: {new_acc.item()*100:.1f}")

        loss = torch.nn.functional.cross_entropy(scores, labels)
        # print(f"loss: {loss.item():.2f}")

        import wandb
        wandb.log({
            "train/acc_emb": original_acc.item(),
            "train/acc": new_acc.item(),
            # "loss": loss.item(),
        })

        return loss