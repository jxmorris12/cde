from typing import Dict, Optional, Union

import copy
import torch
import torch.nn as nn
import transformers

from cde.lib.dist import print0
from cde.lib.tensor import mean_pool, mean_pool_3d

from cde.lib.contextual_bert import ContextualBertModel
from cde.lib.contextual_bert.configuration import ContextualBertConfig


def limit_layers(model: transformers.PreTrainedModel, n_layers: int) -> None:
    if hasattr(model, 'transformer'):
        model.transformer.layer = model.transformer.layer[:n_layers]
    elif hasattr(model, 'encoder'):
        if hasattr(model.encoder, 'layers'):
            model.encoder.layers = model.encoder.layers[:n_layers]
        else:
            model.encoder.layer = model.encoder.layer[:n_layers]
    else:
        raise RuntimeError(f"unknown how to limit layers of model {type(model)}")


def disable_dropout(model: torch.nn.Module):
    dropout_modules = [m for m in model.modules() if isinstance(m, torch.nn.Dropout)]
    for m in dropout_modules:
        m.p = 0.0
    print0(
        f"Disabled {len(dropout_modules)} dropout modules from model type {type(model)}"
    )

class ContextualModelMixin(nn.Module):
    @property
    def num_corpus_tokens(self) -> int:
        return self.transductive_corpus_size * self.transductive_tokens_per_document

    def contextual_init(self):
        self.n_soft_prompt = 8
        self.prompt_projection = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, self.hidden_size * self.n_soft_prompt)
        )
        self.transductive_corpus_size = vars(self.config).get("transductive_corpus_size", 1)
        self.transductive_tokens_per_document = vars(self.config).get("transductive_tokens_per_document", 1)
        self.randomize_dataset_sequence_order = False
        self.sequence_dropout_prob = vars(self.config).get("transductive_sequence_dropout_prob", 0.0)
        if self.sequence_dropout_prob > 0.0:
            self.sequence_dropout_null_embedding = torch.nn.Parameter(
                torch.randn(self.hidden_size) * 0.01,
                requires_grad = True
            )       
        self.output_projection = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, self.hidden_size)
        )

    def _prepare_dataset_embeddings(
            self, 
            input_ids: torch.Tensor, dataset_embeddings: torch.Tensor,
            null_dataset_embedding: bool = False,
        ) -> torch.Tensor:
        if not isinstance(dataset_embeddings, torch.Tensor):
            dataset_embeddings = torch.tensor(
                dataset_embeddings, 
                dtype=torch.float32
            )

        if len(dataset_embeddings.shape) == 2:
            # Auto-expand for a batch.
            dataset_embeddings = dataset_embeddings[None, :, :] # (b, d) -> (1, b, d)
        dataset_embeddings = dataset_embeddings.to(input_ids.device)
        dataset_embeddings = dataset_embeddings.to(torch.float32)
    
        batch_size = input_ids.shape[0]
        if self.transductive_tokens_per_document > 1:
            # Choose N random documents to fill our context window with.
            # This logic is a little confusing but allows us to sample a
            # different batch *per-dataset*
            assert dataset_embeddings.shape[1] == self.transductive_tokens_per_document
            R = torch.randint(
                low=0, 
                high=len(dataset_embeddings), 
                size=(batch_size, self.config.transductive_corpus_size), 
                device=dataset_embeddings.device
            )
            # TODO make this deterministic somehow for evaluation?
            dataset_embeddings = dataset_embeddings[R].reshape((batch_size, self.num_corpus_tokens, self.hidden_size))

        if dataset_embeddings.shape[1] > self.num_corpus_tokens:
            # If too many dataset embeddings are passed in, just take the first N until
            # we have the proper number.
            dataset_embeddings = dataset_embeddings[:, :self.num_corpus_tokens, :]
        
        _, corpus_size, _hidden_size = dataset_embeddings.shape
        if _ == 1:
            # Auto-expand for a batch.
            dataset_embeddings = dataset_embeddings.expand((batch_size, -1, -1))

        if self.training and self.sequence_dropout_prob > 0.0:
            sequence_dropout_mask = (
                torch.rand((batch_size, corpus_size), device=dataset_embeddings.device) < self.sequence_dropout_prob
            )
            null_embeddings = self.sequence_dropout_null_embedding[None, None].expand(batch_size, corpus_size, -1)
            dataset_embeddings = torch.where(
                sequence_dropout_mask[..., None], null_embeddings, dataset_embeddings
            )
        elif null_dataset_embedding:
            null_embeddings = self.sequence_dropout_null_embedding[None, None].expand(batch_size, corpus_size, -1)
            dataset_embeddings = null_embeddings
        
        # backbone_max_seq_length = self.backbone.config.max_trained_positions
        # assert batch_size + (2 * self.n_soft_prompt + corpus_size) <= backbone_max_seq_length, "too many hard negatives for backbone model"
        soft_prompt = torch.ones((1, self.hidden_size), device=dataset_embeddings.device, dtype=torch.float32)
        soft_prompt = self.prompt_projection(soft_prompt).reshape((1, self.n_soft_prompt, self.hidden_size))
        soft_prompt = soft_prompt.expand((len(dataset_embeddings), -1, -1)) # -> (b, 4+b, d) # soft_prompt.repeat((len(input_ids), 1, 1))  
        soft_prompt = torch.cat((dataset_embeddings, soft_prompt), dim=1)

        if self.training and self.randomize_dataset_sequence_order:
            randomized_order = torch.stack(
                [
                    torch.cat(
                        (
                            torch.randperm(corpus_size, device=soft_prompt.device), 
                            torch.arange(self.n_soft_prompt, device=soft_prompt.device) + corpus_size
                        ), dim=0) 
                        for _ in range(batch_size)])
            randomized_order = randomized_order.to(soft_prompt.device)
            soft_prompt = soft_prompt.gather(1, randomized_order[..., None].expand_as(soft_prompt))
        
        return soft_prompt

class BiEncoder(transformers.PreTrainedModel):
    embedder: transformers.PreTrainedModel
    def __init__(
            self, 
            config, #: transformers.PreTrainedConfig, 
            embedder: transformers.PreTrainedModel, 
        ):
        super().__init__(config=config)

        if config.limit_layers:
            print0(f"Limiting layers to {config.limit_layers}")
            limit_layers(embedder, config.limit_layers)
    
        self.embedder = embedder
        # if ("t5" in embedder.config.model_type):
        #     print0(f"using torch.compile() on embedder of type `{embedder.config.model_type}`")
        #     self.embedder = torch.compile(self.embedder) 
        self.hidden_size = self.embedder.config.hidden_size
        # Allow pooling to multiple tokens per document
        self.transductive_tokens_per_document = vars(config).get("transductive_tokens_per_document", 1)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(self.hidden_size, self.config.embedding_output_dim or self.hidden_size),
        )
        self.temp = config.logit_scale

        if config.disable_dropout:
            disable_dropout(self)
        self.pooling_strategy = vars(config).get("pooling_strategy", "mean")

    def forward(
            self, 
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            dataset_input_ids: Optional[torch.Tensor] = None,
            dataset_attention_mask: Optional[torch.Tensor] = None,
            token_type_ids = None,
            output_hidden_states: bool = False,
        ) -> torch.Tensor:
        """
        query_embedding (float torch.Tensor) - shape (batch_size, embedding_dim)
        document_embeddings (float torch.Tensor) - shape (corpus_size, embedding_dim)
            where the corpus_size >= batch_size and is structured like this:
                [d1, d2, d3, hn1_1, hn1_2, hn2_1, hn2_2, hn3_1, hn3_2]
                for a corpus with three documents and two hard negatives per document
        """
        # del dataset_input_ids
        # del dataset_attention_mask
        del token_type_ids

        # from cde.lib.dist import get_rank
        # tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
        # if get_rank() == 0:
        #     breakpoint()
        # torch.distributed.barrier()
        outputs = (
            self.embedder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state
        )

        if self.transductive_tokens_per_document > 1:
            document_embeddings = None
            batch_size, seq_length, output_dim = outputs.shape
            assert seq_length % self.transductive_tokens_per_document == 0 # TODO: Pad to nearest multiple

            outputs = outputs.reshape(
                (batch_size,  self.transductive_tokens_per_document, seq_length // self.transductive_tokens_per_document, output_dim)
            )

            attention_mask = attention_mask.reshape((batch_size, self.transductive_tokens_per_document, -1))
            if self.pooling_strategy == "mean":
                document_embeddings = mean_pool_3d(outputs, attention_mask)
            else:
                document_embeddings = document_embeddings.max(dim=1)
            
            document_embeddings = document_embeddings.reshape((batch_size, self.transductive_tokens_per_document, output_dim))
        else:
            if self.pooling_strategy == "mean":
                document_embeddings = mean_pool(outputs, attention_mask)
            else:
                document_embeddings = document_embeddings.max(dim=1)
        output = self.mlp(document_embeddings)

        if output_hidden_states:
            return {
                "hidden_states": outputs,
                "pooled": output,
            }
        else:
            return output


class TwoEmbeddersWithMLP(transformers.PreTrainedModel):
    embedder: transformers.PreTrainedModel
    embedder: transformers.PreTrainedModel
    dataset_backbone: transformers.PreTrainedModel
    def __init__(
            self, 
            config, #: transformers.PreTrainedConfig, 
            embedder: transformers.PreTrainedModel, 
            dataset_backbone: transformers.PreTrainedModel,
        ):
        super().__init__(config=config)

        if config.limit_layers:
            print0(f"Limiting layers to {config.limit_layers}")
            limit_layers(embedder, config.limit_layers)
            limit_layers(dataset_backbone, config.limit_layers)
        self.embedder = embedder
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

        self.temp = config.logit_scale
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
        
        torch.distributed.barrier()

        dataset_backbone_input_embeddings = mean_pool(
            hidden_states=self.backbone(
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


class DatasetConditionedBiencoder(transformers.PreTrainedModel, ContextualModelMixin):
    def __init__(
            self, 
            config,
            dataset_backbone: transformers.PreTrainedModel,
        ):
        super().__init__(config=config)
        self.backbone = dataset_backbone
        self.hidden_size = self.backbone.config.hidden_size
        self.hidden_size = dataset_backbone.config.hidden_size
        self.input_ln = torch.nn.LayerNorm(
            self.hidden_size, 
            eps=self.backbone.config.layer_norm_epsilon
        )
              
        disable_transductive_rotary_embedding = vars(config).get("disable_transductive_rotary_embedding", True)
        self.contextual_init()
        if self.backbone.config.model_type.startswith("nomic") and disable_transductive_rotary_embedding:
            # We only want to apply positional embeddings to the
            # *text* portion of the backbone network.
            self.backbone.config.rotary_start_pos = 0.0
            rotary_disabled = 0

            rotary_start_pos = self.num_corpus_tokens
            for module in self.backbone.modules():
                if hasattr(module, "rotary_emb_dim"):
                    module.rotary_start_pos = rotary_start_pos
                    rotary_disabled += 1
            print0(f"modified {rotary_disabled} rotary modules – set rotary_start_pos to {rotary_start_pos}")
                
    @property
    def num_corpus_tokens(self) -> int:
        return self.config.transductive_corpus_size * self.transductive_tokens_per_document
    
    def forward(
            self, 
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            dataset_embeddings: torch.Tensor,
            output_hidden_states: bool = False,
            null_dataset_embedding: bool = False,
        ) -> torch.Tensor:
        # print(f"[0] input_ids.shape => {input_ids.shape} // dataset_embeddings.shape =", dataset_embeddings.shape)
        soft_prompt = self._prepare_dataset_embeddings(
            input_ids=input_ids,
            dataset_embeddings=dataset_embeddings,
            null_dataset_embedding=null_dataset_embedding,
        )
        backbone_attention_mask = torch.ones(
            soft_prompt.shape[0:2],
            dtype=torch.long,
            device=soft_prompt.device,
        )
        inputs_embeds = self.backbone.embeddings(input_ids) # (b, s) -> (b, s, d)
        # print("[2] inputs_embeds.shape =", inputs_embeds.shape)
        inputs_embeds = torch.cat((soft_prompt, inputs_embeds), dim=1) # (v, 4+b+s, d)
        # print("[3] inputs_embeds.shape =", inputs_embeds.shape)
        attention_mask = torch.cat((backbone_attention_mask, attention_mask), dim=1)
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
        output = self.output_projection(output_pooled) # (b, 2d) -> (b, d)

        if output_hidden_states:
            return {
                "hidden_states": output_vectors,
                "pooled": output,
            }
        else:
            return output


class DatasetPrefixBiencoder(transformers.PreTrainedModel, ContextualModelMixin):
    def __init__(
            self, 
            config, #: transformers.PreTrainedConfig, 
            embedder: transformers.PreTrainedModel, 
        ):
        super().__init__(config=config)
        self.embedder = embedder
        self.hidden_size = self.embedder.config.hidden_size
        self.contextual_init()
    
    def forward(
            self, 
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            dataset_input_ids: torch.Tensor,
            dataset_attention_mask: torch.Tensor,
            output_hidden_states: bool = False,
        ) -> torch.Tensor:
        R = torch.randint(low=0, high=len(dataset_input_ids), size=(len(input_ids),), device=dataset_input_ids.device)
        
        dataset_input_ids = dataset_input_ids[R]
        input_ids = torch.cat((dataset_input_ids, input_ids), dim=1)

        dataset_attention_mask = torch.ones_like(dataset_attention_mask, device=dataset_attention_mask.device)
        input_attention_mask = torch.cat((dataset_attention_mask, attention_mask), dim=1)
        output_attention_mask = torch.cat(
            (torch.zeros_like(dataset_input_ids), attention_mask), dim=1
        )

        output = self.embedder(
            input_ids=input_ids,
            attention_mask=input_attention_mask,
        ) 
        
        output_vectors = output.last_hidden_state
        output_pooled = mean_pool(output_vectors, output_attention_mask)
        output = self.output_projection(output_pooled) # (b, 2d) -> (b, d)

        if output_hidden_states:
            S_d = dataset_attention_mask.shape[1]
            output_vectors = output_vectors[:, S_d:, :]
            return {
                "hidden_states": output_vectors,
                "pooled": output,
            }
        else:
            return output


class DatasetTransformer(transformers.PreTrainedModel):
    embedder: transformers.PreTrainedModel
    dataset_backbone: transformers.PreTrainedModel
    def __init__(
            self, 
            config,
            embedder: transformers.PreTrainedModel, 
            dataset_backbone: transformers.PreTrainedModel,
        ):
        super().__init__(config=config)

        if config.limit_layers:
            print0(f"Limiting layers to {config.limit_layers}")
            limit_layers(embedder, config.limit_layers)
            limit_layers(dataset_backbone, config.limit_layers)
        
        biencoder_config = copy.deepcopy(config)
        biencoder_config.embedding_output_dim = None
        self.first_stage_model = BiEncoder(
            config=biencoder_config,
            embedder=embedder
        )
        self.second_stage_model = DatasetConditionedBiencoder(
            config=config,
            dataset_backbone=dataset_backbone
        )
        self.temp = config.logit_scale
        if config.disable_dropout:
            disable_dropout(self)
        
        transductive_tie_token_embeddings = vars(self.config).get("transductive_tie_token_embeddings", False)
        if transductive_tie_token_embeddings:
            self.second_stage_model.backbone.embeddings.word_embeddings.weight = (
                self.first_stage_model.embedder.embeddings.word_embeddings.weight
            )

    def forward(
            self, 
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            dataset_input_ids: Optional[torch.Tensor],
            dataset_attention_mask: Optional[torch.Tensor],
            output_hidden_states: bool = False,
        ) -> torch.Tensor:
        """
        input_ids (long torch.Tensor) – ids of input tokens
        attention_mask (bool torch.Tensor)
        """
        dataset_embeddings = self.first_stage_model(
            input_ids=dataset_input_ids, 
            attention_mask=dataset_attention_mask
        )
        return self.second_stage_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            dataset_embeddings=dataset_embeddings,
            output_hidden_states=output_hidden_states,
        )


class ContextualBertWrapper(transformers.PreTrainedModel, ContextualModelMixin):
    def __init__(self, config):
        super().__init__(config=config)
        self.config = config
        # TODO: Save config to disk with a new name to avoid this.
        # second_stage_config = transformers.AutoConfig.from_pretrained(
            # "nomic-ai/nomic-bert-2048", trust_remote_code=True)
        # self.backbone = ContextualBertModel._from_config(second_stage_config)
        self.backbone = ContextualBertModel.from_pretrained("nomic-ai/nomic-bert-2048")
        self.transductive_tokens_per_document = vars(self.config).get("transductive_tokens_per_document", 1)
        self.hidden_size = self.backbone.config.hidden_size
        self.contextual_init()
    
    def forward(
            self, 
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            dataset_embeddings: torch.Tensor,
            output_hidden_states: bool = False,
            null_dataset_embedding: bool = False,
        ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        encoder_hidden_states = self._prepare_dataset_embeddings(
            input_ids=input_ids,    
            dataset_embeddings=dataset_embeddings,
            null_dataset_embedding=null_dataset_embedding,
        )
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
        )
        output_vectors = outputs.last_hidden_state
        output_pooled = mean_pool(output_vectors, attention_mask)
        pooled_output = self.output_projection(output_pooled) # (b, 2d) -> (b, d)

        if output_hidden_states:
            return {
                "hidden_states": output_vectors,
                "pooled": pooled_output,
            }
        else:
            return pooled_output


class ContextualCrossAttention(transformers.PreTrainedModel):
    embedder: transformers.PreTrainedModel
    dataset_backbone: transformers.PreTrainedModel
    def __init__(
            self, 
            config,
            embedder: transformers.PreTrainedModel, 
        ):
        super().__init__(config=config)

        if config.limit_layers:
            print0(f"Limiting layers to {config.limit_layers}")
            limit_layers(embedder, config.limit_layers)
        
        biencoder_config = copy.deepcopy(config)
        biencoder_config.embedding_output_dim = None
        self.first_stage_model = BiEncoder(
            config=biencoder_config,
            embedder=embedder
        )
        self.temp = config.logit_scale
        if config.disable_dropout:
            disable_dropout(self)
        self.hidden_size = self.first_stage_model.embedder.config.hidden_size
        self.second_stage_model = ContextualBertWrapper(config=config)
    
    def forward(
            self, 
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            dataset_input_ids: Optional[torch.Tensor],
            dataset_attention_mask: Optional[torch.Tensor],
            output_hidden_states: bool = False,
        ) -> torch.Tensor:
        """
        input_ids (long torch.Tensor) – ids of input tokens
        attention_mask (bool torch.Tensor)
        """
        dataset_embeddings = self.first_stage_model(
            input_ids=dataset_input_ids, 
            attention_mask=dataset_attention_mask
        )
        return self.second_stage_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            dataset_embeddings=dataset_embeddings,
            output_hidden_states=output_hidden_states,
        )


def get_model_class(name: str):
    if name == 'two_head_mlp':
        return TwoEmbeddersWithMLP
    elif name in 'transductive': 
        return DatasetTransformer
    elif name == 'biencoder':
        return BiEncoder
    elif name == "dataset_prefix_biencoder":
        return DatasetPrefixBiencoder
    elif name == "contextual_cross_attention":
        return ContextualCrossAttention
    else:
        raise ValueError(f'unknown model cls {name}')
