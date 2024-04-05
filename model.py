from typing import Optional

import torch
import transformers

from lib.tensor import mean_pool


def limit_layers(model: transformers.PreTrainedModel, n_layers: int) -> None:
    if hasattr(model, 'transformer'):
        model.transformer.layer = model.transformer.layer[:n_layers]
    elif hasattr(model, 'encoder'):
        model.encoder.layer = model.encoder.layer[:n_layers]
    else:
        raise RuntimeError(f"unknown how to limit layers of model {type(model)}")


def disable_dropout(model: torch.nn.Module):
    dropout_modules = [m for m in model.modules() if isinstance(m, torch.nn.Dropout)]
    for m in dropout_modules:
        m.p = 0.0
    print(
        f"Disabled {len(dropout_modules)} dropout modules from model type {type(model)}"
    )


class TwoEmbeddersWithMLP(transformers.PreTrainedModel):
    embedder: transformers.PreTrainedModel
    dataset_embedder: transformers.PreTrainedModel
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
        # self.tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased')

        if config.limit_layers:
            print(f"Limiting layers to {config.limit_layers}")
            limit_layers(embedder, config.limit_layers)
            limit_layers(dataset_embedder, config.limit_layers)
            limit_layers(dataset_backbone, config.limit_layers)
        self.embedder = embedder
        self.dataset_embedder = dataset_embedder
        if hasattr(dataset_embedder, "encoder"):
            self.dataset_embedder = dataset_embedder.encoder
        else:
            self.dataset_embedder = dataset_embedder
        self.backbone = dataset_backbone

        # TODO make this a little nicer. (Not every model has 'embeddings...')
        self.backbone = dataset_backbone
        self.backbone.embeddings.word_embeddings.weight.requires_grad = False
        self.backbone.embeddings.word_embeddings.weight.fill_(0.0)

        self.hidden_size = self.embedder.config.hidden_size
        joint_hidden_size = (self.embedder.config.hidden_size + self.backbone.config.hidden_size)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(joint_hidden_size, joint_hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(joint_hidden_size, self.hidden_size * 2),
            torch.nn.GELU(),
            torch.nn.Linear(self.hidden_size * 2, self.hidden_size),
        )

        self.temp = config.contrastive_temp
        if config.disable_dropout:
            disable_dropout(self)
    
    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            dataset_input_ids: torch.Tensor,
            dataset_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = input_ids.shape[0]
        _corpus_size = dataset_input_ids.shape[0]
        dataset_backbone_input_embeddings = mean_pool(
            hidden_states=self.dataset_embedder(
                input_ids=dataset_input_ids,
                attention_mask=dataset_attention_mask
            ).last_hidden_state,
            attention_mask=dataset_attention_mask,
        )
        emb_dtype = self.backbone.embeddings.word_embeddings.weight.dtype
        dataset_backbone_input_embeddings = dataset_backbone_input_embeddings.to(emb_dtype)
        assert len(dataset_backbone_input_embeddings.shape) == 2 # (b, d)
        dataset_backbone_input_embeddings = dataset_backbone_input_embeddings[:, None, :]
        dataset_backbone_attention_mask = torch.ones(
            dataset_backbone_input_embeddings.shape[0:2], 
            device=dataset_input_ids.device,
            dtype=torch.long
        )
        dataset_intermediate_embeddings = self.backbone(
            inputs_embeds=dataset_backbone_input_embeddings,
            attention_mask=dataset_backbone_attention_mask,
        ).last_hidden_state
        dataset_embedding = dataset_intermediate_embeddings[:, 0, :].mean(
            dim=0, 
            keepdim=True
        )
        embeddings = mean_pool(
            hidden_states=self.embedder(
                input_ids=input_ids,
                attention_mask=attention_mask
            ).last_hidden_state,
            attention_mask=attention_mask,
        )
        assert dataset_embedding.shape[0] == 1 # (1, d)
        assert embeddings.shape[0] == batch_size # (b, d)
        mlp_input_embeddings = torch.cat(
            (
                dataset_embedding.expand(batch_size, -1),
                embeddings
            ),
            dim=1
        )
        outputs = self.mlp(mlp_input_embeddings)
        assert outputs.shape == (batch_size, self.hidden_size)
        return outputs


class DatasetTransformer(transformers.PreTrainedModel):
    embedder: transformers.PreTrainedModel
    dataset_embedder: transformers.PreTrainedModel
    dataset_backbone: transformers.PreTrainedModel
    def __init__(
            self, 
            config, #: transformers.PreTrainedConfig, 
            embedder: transformers.PreTrainedModel, 
            dataset_backbone: transformers.PreTrainedModel,
            dataset_embedder: transformers.PreTrainedModel = None,  #  ignored 
        ):
        super().__init__(config=config)

        if config.limit_layers:
            print(f"Limiting layers to {config.limit_layers}")
            limit_layers(embedder, config.limit_layers)
            limit_layers(dataset_backbone, config.limit_layers)
        
        del dataset_embedder
        self.embedder = embedder
        self.backbone = dataset_backbone

        self.hidden_size = self.embedder.config.hidden_size

        self.backbone = dataset_backbone
        self.embedding_dim = self.embedder.config.hidden_size
        self.hidden_size = self.backbone.config.hidden_size
        self.n_soft_prompt = 8

        if self.backbone.config.model_type.startswith("nomic"):
            # We only want to apply positional embeddings to the
            # *text* portion of the backbone network.
            self.backbone.config.rotary_start_pos = 0.0
            rotary_disabled = 0
            for module in self.backbone.modules():
                if hasattr(module, "rotary_emb_dim"):
                    module.rotary_start_pos = config.corpus_size
            print(f"modified {rotary_disabled} rotary modules – set rotary_start_pos to {config.corpus_size}")

        # TODO: Argparse, ablate, and potentially remove the soft prompt portion
        # of this model.
        self.prompt_projection = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_dim, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, self.hidden_size * self.n_soft_prompt)
        )
        self.output_projection = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, self.hidden_size)
        )
        self.corpus_projection = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_dim, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, self.hidden_size)
        )
        self.temp = config.contrastive_temp
        if config.disable_dropout:
            disable_dropout(self)
        
        self.randomize_dataset_sequence_order = True
        
    def forward_first_stage(
            self,
            dataset_input_ids: torch.Tensor,
            dataset_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # print("forward_first_stage //", dataset_input_ids.shape, dataset_attention_mask.shape)
        outputs = (
            self.embedder(
                input_ids=dataset_input_ids,
                attention_mask=dataset_attention_mask).last_hidden_state
        )
        dataset_embeddings = mean_pool(outputs, dataset_attention_mask) # (b, s, d) -> (b, d)
        return self.corpus_projection(dataset_embeddings) # (1, b, d) -> (1, b, d)
        
    def forward_second_stage(
            self, 
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            dataset_embeddings: torch.Tensor,
        ) -> torch.Tensor:
        # print("forward_second_stage //", input_ids.shape, input_ids.shape, "//", dataset_embeddings.shape)
        dataset_embeddings = dataset_embeddings[None, :, :] # (b, d) -> (1, b, d)
        
        _batch_size = input_ids.shape[0]
        _, corpus_size, _hidden_dim = dataset_embeddings.shape
        assert _ == 1
        
        # backbone_max_seq_length = self.backbone.config.max_trained_positions
        # assert batch_size + (2 * self.n_soft_prompt + corpus_size) <= backbone_max_seq_length, "too many hard negatives for backbone model"

        soft_prompt = torch.ones((1, self.embedding_dim), device=dataset_embeddings.device, dtype=torch.float32)
        soft_prompt = self.prompt_projection(soft_prompt).reshape((1, self.n_soft_prompt, self.hidden_size))
        soft_prompt = torch.cat((dataset_embeddings, soft_prompt), dim=1)
        soft_prompt = soft_prompt.expand((len(input_ids), -1, -1)) # -> (b, 4+b, d) # soft_prompt.repeat((len(input_ids), 1, 1))  
        inputs_embeds = self.backbone.embeddings(input_ids) # (b, s) -> (b, s, d)
        inputs_embeds = torch.cat((soft_prompt, inputs_embeds), dim=1) # (v, 4+b+s, d)

        backbone_attention_mask = torch.ones(
            soft_prompt.shape[0:2],
            dtype=torch.long,
            device=soft_prompt.device,
        )
        attention_mask = torch.cat((backbone_attention_mask, attention_mask), dim=1)
        # print("? calling backbone with ", inputs_embeds.shape, "/", inputs_embeds.norm(p=2))
        output = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        ) # (1, 4 + b + s, d)
        # trim soft prompt
        output_vectors = output.last_hidden_state

        # use only these tokens
        n_soft_prompt_tokens = soft_prompt.shape[1]

        output_vectors = output.last_hidden_state[:, n_soft_prompt_tokens:, :]
        output_attention_mask = attention_mask[:, n_soft_prompt_tokens:]
        output_pooled = mean_pool(output_vectors, output_attention_mask)

        # average with original vectors
        # TODO: Argparse for pooling strategy.
        # output_vectors = torch.cat((soft_prompt_pooled, output_pooled), dim=1) # (b, d) + (b, d) -> (b, 2d)
        output_vectors = output_pooled
        return self.output_projection(output_vectors) # (b, 2d) -> (b, d)

    def forward(
            self, 
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            dataset_input_ids: Optional[torch.Tensor] = None,
            dataset_attention_mask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
        """
        input_ids (long torch.Tensor) – ids of input tokens
        attention_mask (bool torch.Tensor)
        """
        dataset_embeddings = self.forward_first_stage(
            dataset_input_ids=dataset_input_ids, 
            dataset_attention_mask=dataset_attention_mask
        )
        return self.forward_second_stage(
            input_ids=input_ids,
            attention_mask=attention_mask,
            dataset_embeddings=dataset_embeddings,
        )


class BiEncoder(transformers.PreTrainedModel):
    embedder: transformers.PreTrainedModel
    def __init__(
            self, 
            config, #: transformers.PreTrainedConfig, 
            embedder: transformers.PreTrainedModel, 
            dataset_embedder: transformers.PreTrainedModel, 
            dataset_backbone: transformers.PreTrainedModel,
        ):
        super().__init__(config=config)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased')

        if config.limit_layers:
            print(f"Limiting layers to {config.limit_layers}")
            limit_layers(embedder, config.limit_layers)
            limit_layers(dataset_backbone, config.limit_layers)
        
        del dataset_embedder
        del dataset_backbone
        self.embedder = embedder
        self.hidden_size = self.embedder.config.hidden_size
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.temp = config.contrastive_temp

        if config.disable_dropout:
            disable_dropout(self)

        self.embedding_dim = self.embedder.config.hidden_size

    def forward(
            self, 
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            dataset_input_ids: torch.Tensor,
            dataset_attention_mask: torch.Tensor,
        ) -> torch.Tensor:
        """
        query_embedding (float torch.Tensor) - shape (batch_size, embedding_dim)
        document_embeddings (float torch.Tensor) - shape (corpus_size, embedding_dim)
            where the corpus_size >= batch_size and is structured like this:
                [d1, d2, d3, hn1_1, hn1_2, hn2_1, hn2_2, hn3_1, hn3_2]
                for a corpus with three documents and two hard negatives per document
        """
        del dataset_input_ids
        del dataset_attention_mask

        outputs = (
            self.embedder(
                input_ids=input_ids,
                attention_mask=attention_mask).last_hidden_state
        )
        document_embeddings = mean_pool(outputs, attention_mask)
        # return
        return self.mlp(document_embeddings)


def get_model_class(name: str):
    if name == 'two_head_mlp':
        return TwoEmbeddersWithMLP
    elif name == 'query_independent_dt':
        return DatasetTransformer
    elif name == 'biencoder':
        return BiEncoder
    else:
        raise ValueError(f'unknown model cls {name}')