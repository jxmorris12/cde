from typing import Optional

import dataclasses
import torch

@dataclasses.dataclass
class FakeConfig:
    hidden_size = 32
    model_type = "fake"
    transductive_corpus_size = 20

class FakeModelOutput:
    def __init__(self, last_hidden_state: torch.Tensor):
        self.last_hidden_state = last_hidden_state

class FakeEmbedder(torch.nn.Module):
    def __init__(self, *args, **kwargs):
          self.config = FakeConfig()
          super().__init__()

    def embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length = input_ids.shape
        return torch.randn(
            batch_size, seq_length, self.config.hidden_size, device=input_ids.device)

    def forward(self, input_ids: torch.Tensor = None, attention_mask: torch.Tensor = None, inputs_embeds: torch.Tensor = None) -> FakeModelOutput:
        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
            device = input_ids.device
        else:
            batch_size, seq_length = inputs_embeds.shape[0:2]
            device = inputs_embeds.device
        last_hidden_state = torch.randn(
            batch_size, seq_length, self.config.hidden_size, device=device)
        return FakeModelOutput(
            last_hidden_state=last_hidden_state
        )
    

class FakeDatasetTransformer(torch.nn.Module):
    def __init__(self, *args, **kwargs):
          self.config = FakeConfig()
          super().__init__()

    def first_stage_model(
            self, 
            input_ids: torch.Tensor, 
            attention_mask: torch.Tensor
        ) -> torch.Tensor:
        batch_size, seq_length = input_ids.shape
        return torch.randn(
            batch_size, seq_length, self.config.hidden_size, device=input_ids.device)
    
    def second_stage_model(
            self, 
            input_ids: torch.Tensor, 
            attention_mask: torch.Tensor, 
            dataset_embeddings: torch.Tensor
        ) -> torch.Tensor:
        batch_size, seq_length = input_ids.shape
        return torch.randn(
            batch_size, seq_length, self.config.hidden_size, device=input_ids.device)

    def forward(
            self, 
            input_ids: torch.Tensor, 
            attention_mask: torch.Tensor, 
            dataset_input_ids: Optional[torch.Tensor],
            dataset_attention_mask: Optional[torch.Tensor],
        ) -> FakeModelOutput:
        dataset_embeddings = self.first_stage_model(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        return self.second_stage_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            dataset_embeddings=dataset_embeddings,
        )