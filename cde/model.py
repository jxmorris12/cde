from typing import Optional

import copy
import torch
import torch.nn as nn
import transformers

from cde.lib.dist import print0
from cde.lib.tensor import mean_pool, mean_pool_3d, mean_pool_weighted, last_token_pool

from cde.lib import load_embedder_and_tokenizer, ContextualModelConfig


gpt_tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")

def limit_layers(model: transformers.PreTrainedModel, n_layers: int) -> None:
    if hasattr(model, 'transformer'):
        if hasattr(model.transformer, 'h'):
            # gpt2
            model.transformer.h = model.transformer.h[:n_layers]
        else:
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


def disable_causality(model: torch.nn.Module):
    disabled_modules = 0
    for m in model.modules():
        if hasattr(m, "is_causal"):
            m.is_causal = False
            disabled_modules += 1
    print0(
        f"Set is_causal=False in {disabled_modules} modules from model type {type(model)}"
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
        self.randomize_dataset_sequence_order = True
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
            dataset_embeddings = torch.tensor(dataset_embeddings)

        if len(dataset_embeddings.shape) == 2:
            # Auto-expand for a batch.
            dataset_embeddings = dataset_embeddings[None, :, :] # (b, d) -> (1, b, d)
        dataset_embeddings = dataset_embeddings.to(input_ids.device)
    
        batch_size = input_ids.shape[0]
        if (self.transductive_tokens_per_document > 1):
            if self.training:
                # Choose N random documents to fill our context window with.
                # This logic is a little confusing but allows us to sample a
                # different batch *per-document*
                assert dataset_embeddings.shape[1] == self.transductive_tokens_per_document
                R = torch.randint(
                    low=0, 
                    high=len(dataset_embeddings), 
                    size=(batch_size, self.config.transductive_corpus_size), 
                    device=dataset_embeddings.device
                )
                # TODO make this deterministic somehow for evaluation?
                dataset_embeddings = dataset_embeddings[R].reshape((batch_size, self.num_corpus_tokens, self.hidden_size))
            else:
                dataset_embeddings = dataset_embeddings.reshape((1, self.num_corpus_tokens, self.hidden_size))
                # print("reshaped to dataset_embeddings.shape =", dataset_embeddings.shape)

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
        
        # print(f"[ContextualModelMixin] dataset_embeddings.shape = {dataset_embeddings.shape}")
        
        # backbone_max_seq_length = self.backbone.config.max_trained_positions
        # assert batch_size + (2 * self.n_soft_prompt + corpus_size) <= backbone_max_seq_length, "too many hard negatives for backbone model"
        soft_prompt = torch.ones((1, self.hidden_size), device=dataset_embeddings.device, dtype=dataset_embeddings.dtype)
        soft_prompt = self.prompt_projection(soft_prompt).reshape((1, self.n_soft_prompt, self.hidden_size))
        soft_prompt = soft_prompt.expand((len(dataset_embeddings), -1, -1)) # -> (b, 4+b, d) # soft_prompt.repeat((len(input_ids), 1, 1))  
        soft_prompt = torch.cat((dataset_embeddings, soft_prompt), dim=1)

        # print(f"[ContextualModelMixin] soft_prompt.shape = {soft_prompt.shape}")

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
        ):
        super().__init__(config=config)
        embedder, _ = load_embedder_and_tokenizer(
            config.embedder,
        )

        if config.limit_layers:
            print0(f"Limiting layers to {config.limit_layers}")
            limit_layers(embedder, config.limit_layers)
    
        self.embedder = embedder
        # if ("t5" in embedder.config.model_type):
        #     print0(f"using torch.compile() on embedder of type `{embedder.config.model_type}`")
        #     self.embedder = torch.compile(self.embedder) 
        self.hidden_size = self.embedder.config.hidden_size
        # Allow pooling to multiple tokens per document
        self.transductive_tokens_per_document = vars(self.config).get("transductive_tokens_per_document", 1)
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

            if seq_length % self.transductive_tokens_per_document != 0:
                # Pad to nearest multiple
                n_extra_embeds = self.transductive_tokens_per_document - (seq_length % self.transductive_tokens_per_document)
                outputs = torch.cat(
                    (outputs, torch.zeros((batch_size, n_extra_embeds, output_dim), device=outputs.device)),
                    dim=1
                )
                attention_mask = torch.cat(
                    (attention_mask, torch.zeros((batch_size, n_extra_embeds), device=attention_mask.device)),
                    dim=1
                )
                seq_length += n_extra_embeds
                print(f"Added {n_extra_embeds} padding tokens to input_ids and attention_mask")
            
            # print("ftransductive_tokens_per_document {self.transductive_tokens_per_document} outputs.shape =", outputs.shape)

            outputs = outputs.reshape(
                (batch_size,  self.transductive_tokens_per_document, seq_length // self.transductive_tokens_per_document, output_dim)
            )

            attention_mask = attention_mask.reshape((batch_size, self.transductive_tokens_per_document, -1))
            document_embeddings = mean_pool_3d(outputs, attention_mask)
            
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


class DatasetConditionedAutoregressive(transformers.PreTrainedModel, ContextualModelMixin):
    def __init__(
            self, 
            config,
            dataset_backbone: transformers.PreTrainedModel,
            first_stage_hidden_size: int,
        ):
        super().__init__(config=config)
        self.backbone = dataset_backbone
        self.backbone_hidden_size = self.backbone.config.hidden_size
        self.hidden_size = first_stage_hidden_size # Input token size
        self.contextual_init()
        disable_causality(self.backbone)
        
        self.input_ln = torch.nn.LayerNorm(
            self.backbone_hidden_size, 
            eps=1e-5
        )

        self.pool_ignore_contextual_tokens = vars(self.config).get("pool_ignore_contextual_tokens", False)
        self.pool_ignore_instruction_tokens = vars(self.config).get("pool_ignore_instruction_tokens", False)
        self.pool_instruction_end_id = self.backbone.config.bos_token_id
        
        # Override contextual init
        self.output_projection = torch.nn.Sequential(
            torch.nn.Linear(self.backbone_hidden_size, self.backbone_hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.backbone_hidden_size, self.backbone_hidden_size)
        )
        self._shift_rotary_embedding()
                
    @property
    def num_corpus_tokens(self) -> int:
        return self.config.transductive_corpus_size * self.transductive_tokens_per_document

    @property
    def corpus_token_ratio(self) -> float:
        # How many tokens from the first stage make one token in the second
        # stage?
        return self.backbone_hidden_size / self.hidden_size
    
    def corpus_token_pad_size(self, n_tokens: int) -> int:
        return self.hidden_size % self.backbone_hidden_size
    
    def _shift_rotary_embedding(self) -> None:
        disable_transductive_rotary_embedding = vars(self.config).get("disable_transductive_rotary_embedding", True)
        # TODO: Can we do this for LLAMA?
        print0("Warning: Positional embedding disabling not implemented for LLAMA.")
    
    def forward(
            self, 
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            dataset_embeddings: torch.Tensor,
            output_hidden_states: bool = False,
            null_dataset_embedding: bool = False,
        ) -> torch.Tensor:
        soft_prompt = self._prepare_dataset_embeddings(
            input_ids=input_ids,
            dataset_embeddings=dataset_embeddings,
            null_dataset_embedding=null_dataset_embedding,
        )
        
        # Reshape for this model.
        # print("[DatasetConditionedAutoregressive] 1 -> soft_prompt.shape =", soft_prompt.shape)
        num_soft_elements = torch.prod(torch.tensor(soft_prompt.shape[1:])).item()
        soft_prompt = soft_prompt.reshape((soft_prompt.shape[0], num_soft_elements))
        num_padding_elements = self.backbone_hidden_size - (num_soft_elements % self.backbone_hidden_size)
        padding = torch.ones((soft_prompt.shape[0], num_padding_elements), device=soft_prompt.device)
        soft_prompt = torch.cat((soft_prompt, padding), dim=1)
        soft_prompt = soft_prompt.reshape(
            (soft_prompt.shape[0], -1, self.backbone_hidden_size)
        )
        soft_prompt = self.input_ln(soft_prompt)
        # print("[DatasetConditionedAutoregressive] 2 -> soft_prompt.shape =", soft_prompt.shape)

        backbone_attention_mask = torch.ones(
            soft_prompt.shape[0:2],
            dtype=torch.long,
            device=soft_prompt.device,
        )
        token_embeddings = self.backbone.get_input_embeddings()
        inputs_embeds = token_embeddings(input_ids) # (b, s) -> (b, s, d)
        # print("[2] inputs_embeds.shape =", inputs_embeds.shape)
        inputs_embeds = torch.cat((soft_prompt, inputs_embeds), dim=1) # (v, 4+b+s, d)
        # print("[3.a] inputs_embeds.shape =", inputs_embeds.shape)
        input_attention_mask = torch.cat((backbone_attention_mask, attention_mask), dim=1)
        # print("[3.b] attention_mask.shape =", attention_mask.shape)

        output = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=input_attention_mask,
            output_hidden_states=True,
        ) # (1, 4 + b + s, d)
        # trim soft prompt
        output_vectors = output.hidden_states[-1]
        n_soft_prompt_tokens = soft_prompt.shape[1]

        if self.pool_ignore_instruction_tokens:
            # Denote the end of an instruction with an extra BOS token.
            # This is a bit arcane but relies on the fact that there will be a BOS token after the
            # instruction, but also there may or may not be a BOS token at the beginning.
            instruction_end_idx = (
                (input_ids == self.pool_instruction_end_id) & 
                attention_mask &
                (torch.arange(input_ids.shape[1], device=input_ids.device)[None, :] > 0)
            ).int().argmax(1)
            is_instruction_token_mask = (
                torch.arange(input_ids.shape[1], device=input_ids.device)[None, :] <= instruction_end_idx[:, None]
            )
            # catch edge case where there is no instruction
            is_instruction_token_mask = is_instruction_token_mask.where(
                (instruction_end_idx > 0)[:, None], torch.zeros_like(is_instruction_token_mask)
            )
            input_attention_mask = torch.cat((
                backbone_attention_mask, 
                attention_mask & ~is_instruction_token_mask), dim=1
            )

        output_attention_mask = input_attention_mask
        if self.pool_ignore_contextual_tokens:
            output_vectors = output_vectors[:, n_soft_prompt_tokens:, :]
            output_attention_mask = output_attention_mask[:, n_soft_prompt_tokens:]

        # Take last token position
        if vars(self.config).get("pooling_strategy") == "last_token":
            output_pooled = last_token_pool(output_vectors, output_attention_mask)
        elif vars(self.config).get("pooling_strategy") == "mean":
            output_pooled = mean_pool(output_vectors, output_attention_mask)
        else:
            output_pooled = mean_pool_weighted(output_vectors, output_attention_mask)

        # average with original vectors
        output = self.output_projection(output_pooled) # (b, 2d) -> (b, d)

        if output_hidden_states:
            return {
                "hidden_states": output_vectors,
                "pooled": output,
            }
        else:
            return output


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
        self.contextual_init()
        self._shift_rotary_embedding()

        self.pool_ignore_contextual_tokens = vars(self.config).get("pool_ignore_contextual_tokens", False)
        self.pool_ignore_instruction_tokens = vars(self.config).get("pool_ignore_instruction_tokens", False)
                
    @property
    def num_corpus_tokens(self) -> int:
        return self.config.transductive_corpus_size * self.transductive_tokens_per_document
    
    def _shift_rotary_embedding(self) -> None:
        disable_transductive_rotary_embedding = vars(self.config).get("disable_transductive_rotary_embedding", True)
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
    
    def forward(
            self, 
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            dataset_embeddings: torch.Tensor,
            output_hidden_states: bool = False,
            null_dataset_embedding: bool = False,
        ) -> torch.Tensor:
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
        inputs_embeds = torch.cat((soft_prompt, inputs_embeds), dim=1) # (v, 4+b+s, d)
        attention_mask = torch.cat((backbone_attention_mask, attention_mask), dim=1)
        output = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        ) # (1, 4 + b + s, d)
        # trim soft prompt
        output_vectors = output.last_hidden_state

        # use only these tokens
        n_soft_prompt_tokens = soft_prompt.shape[1]

        if self.pool_ignore_instruction_tokens:
            # Denote the end of an instruction with an extra BOS token.
            # This is a bit arcane but relies on the fact that there will be a BOS token after the
            # instruction, but also there may or may not be a BOS token at the beginning.
            instruction_end_idx = (
                (input_ids == self.pool_instruction_end_id) & 
                attention_mask &
                (torch.arange(input_ids.shape[1], device=input_ids.device)[None, :] > 0)
            ).int().argmax(1)
            is_instruction_token_mask = (
                torch.arange(input_ids.shape[1], device=input_ids.device)[None, :] <= instruction_end_idx[:, None]
            )
            # catch edge case where there is no instruction
            is_instruction_token_mask = is_instruction_token_mask.where(
                (instruction_end_idx > 0)[:, None], torch.zeros_like(is_instruction_token_mask)
            )
            input_attention_mask = torch.cat((backbone_attention_mask, attention_mask & ~is_instruction_token_mask), dim=1)

        output_attention_mask = input_attention_mask
        if self.pool_ignore_contextual_tokens:
            output_vectors = output_vectors[:, n_soft_prompt_tokens:, :]
            output_attention_mask = output_attention_mask[:, n_soft_prompt_tokens:]
        output_pooled = mean_pool(output_vectors, output_attention_mask)

        # average with original vectors
        # TODO: Argparse for pooling strategy.
        # output_vectors = torch.cat((soft_prompt_pooled, output_pooled), dim=1) # (b, d) + (b, d) -> (b, 2d)
        # print("output_pooled.shape =", output_pooled.shape)
        output = self.output_projection(output_pooled) # (b, 2d) -> (b, d)

        # print("returning output.shape =", output.shape)

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
    config_class = ContextualModelConfig
    embedder: transformers.PreTrainedModel
    dataset_backbone: transformers.PreTrainedModel
    def __init__(
            self, 
            config,
        ):
        super().__init__(config=config)
        dataset_backbone, _ = load_embedder_and_tokenizer(
            vars(config).get("dataset_backbone") or config.embedder
        )

        if config.limit_layers:
            print0(f"Limiting layers to {config.limit_layers}")
            limit_layers(dataset_backbone, config.limit_layers)
        
        biencoder_config = copy.deepcopy(config)
        biencoder_config.embedding_output_dim = None
        biencoder_config.limit_layers = vars(self.config).get("limit_layers_first_stage", None)
        self.first_stage_model = BiEncoder(
            config=biencoder_config,
        )

        if vars(config).get("autoregressive_backbone", False):
            self.second_stage_model = DatasetConditionedAutoregressive(
                config=config,
                dataset_backbone=dataset_backbone,
                first_stage_hidden_size=self.first_stage_model.hidden_size,
            )
        else:
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



def get_model_class(name: str):
    if name in 'transductive': 
        return DatasetTransformer
    elif name == 'biencoder':
        return BiEncoder
    elif name == "biencoder_plus_plus":
        from cde.model_extra import BiEncoderPlusPlus
        return BiEncoderPlusPlus
    elif name == "dataset_prefix_biencoder":
        return DatasetPrefixBiencoder
    else:
        raise ValueError(f'unknown model cls {name}')
