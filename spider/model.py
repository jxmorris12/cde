from typing import Optional

import copy
import torch
import transformers

from spider.lib.dist import print0
from spider.lib.tensor import mean_pool


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


class BiEncoder(transformers.PreTrainedModel):
    embedder: transformers.PreTrainedModel
    def __init__(
            self, 
            config, #: transformers.PreTrainedConfig, 
            embedder: transformers.PreTrainedModel, 
        ):
        super().__init__(config=config)

        if config.limit_layers:
            print(f"Limiting layers to {config.limit_layers}")
            limit_layers(embedder, config.limit_layers)
    
        self.embedder = embedder
        # if ("t5" in embedder.config.model_type):
        #     print(f"using torch.compile() on embedder of type `{embedder.config.model_type}`")
        #     self.embedder = torch.compile(self.embedder) 
        self.hidden_size = self.embedder.config.hidden_size
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(self.hidden_size, self.config.embedding_output_dim or self.hidden_size),
        )
        self.temp = config.logit_scale

        if config.disable_dropout:
            disable_dropout(self)

        self.embedding_dim = self.embedder.config.hidden_size

    def forward(
            self, 
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            dataset_input_ids: Optional[torch.Tensor] = None,
            dataset_attention_mask: Optional[torch.Tensor] = None,
            token_type_ids = None,
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
        del token_type_ids
        outputs = (
            self.embedder(
                input_ids=input_ids,
                attention_mask=attention_mask).last_hidden_state
        )
        document_embeddings = mean_pool(outputs, attention_mask)
        # return
        output = self.mlp(document_embeddings)
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


class DatasetConditionedBiencoder(transformers.PreTrainedModel):
    def __init__(
            self, 
            config,
            dataset_backbone: transformers.PreTrainedModel,
        ):
        super().__init__(config=config)
        self.backbone = dataset_backbone
        self.embedding_dim = dataset_backbone.config.hidden_size
        self.hidden_size = self.backbone.config.hidden_size
        self.n_soft_prompt = 8
        self.hidden_size = dataset_backbone.config.hidden_size

        disable_transductive_rotary_embedding = vars(config).get("disable_transductive_rotary_embedding", True)
        if self.backbone.config.model_type.startswith("nomic") and disable_transductive_rotary_embedding:
            # We only want to apply positional embeddings to the
            # *text* portion of the backbone network.
            self.backbone.config.rotary_start_pos = 0.0
            rotary_disabled = 0
            for module in self.backbone.modules():
                if hasattr(module, "rotary_emb_dim"):
                    module.rotary_start_pos = config.transductive_corpus_size
                    rotary_disabled += 1
            print0(f"modified {rotary_disabled} rotary modules – set rotary_start_pos to {config.transductive_corpus_size}")

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
            torch.nn.Linear(self.hidden_size, self.config.embedding_output_dim or self.hidden_size)
        )
        self.randomize_dataset_sequence_order = True
        self.sequence_dropout_prob = vars(config).get("transductive_sequence_dropout_prob", 0.0)
        if self.sequence_dropout_prob > 0.0:
            self.sequence_dropout_null_embedding = torch.nn.Parameter(
                torch.randn(self.embedding_dim) * 0.01,
                requires_grad = True
            )
    
    def forward(
            self, 
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            dataset_embeddings: torch.Tensor,
            null_dataset_embedding: bool = False
        ) -> torch.Tensor:
        dataset_embeddings = dataset_embeddings[None, :, :] # (b, d) -> (1, b, d)
        dataset_embeddings = dataset_embeddings.to(input_ids.device)

        if dataset_embeddings.shape[1] > self.config.transductive_corpus_size:
            # If too many dataset embeddings are passed in, just take the first N until
            # we have the proper number.
            dataset_embeddings = dataset_embeddings[:, :self.config.transductive_corpus_size, :]
        
        batch_size = input_ids.shape[0]
        _, corpus_size, _hidden_dim = dataset_embeddings.shape
        assert _ == 1

        dataset_embeddings = dataset_embeddings.expand((batch_size, -1, -1)) # -> (b, 4+b, d) # soft_prompt.repeat((len(input_ids), 1, 1))  
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
        soft_prompt = torch.ones((1, self.embedding_dim), device=dataset_embeddings.device, dtype=torch.float32)
        soft_prompt = self.prompt_projection(soft_prompt).reshape((1, self.n_soft_prompt, self.hidden_size))
        soft_prompt = soft_prompt.expand((batch_size, -1, -1)) # -> (b, 4+b, d) # soft_prompt.repeat((len(input_ids), 1, 1))  
        soft_prompt = torch.cat((dataset_embeddings, soft_prompt), dim=1)

        if self.training and self.randomize_dataset_sequence_order:
            randomized_order = torch.stack(
                [
                    torch.cat(
                        (torch.randperm(corpus_size), 
                        torch.arange(self.n_soft_prompt) + corpus_size), dim=0) 
                        for _ in range(batch_size)])
            randomized_order = randomized_order.to(soft_prompt.device)
            soft_prompt = soft_prompt.gather(1, randomized_order[..., None].expand_as(soft_prompt))
        
        inputs_embeds = self.backbone.embeddings(input_ids) # (b, s) -> (b, s, d)
        inputs_embeds = torch.cat((soft_prompt, inputs_embeds), dim=1) # (v, 4+b+s, d)

        backbone_attention_mask = torch.ones(
            soft_prompt.shape[0:2],
            dtype=torch.long,
            device=soft_prompt.device,
        )
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
        output_vectors = output_pooled
        output = self.output_projection(output_vectors) # (b, 2d) -> (b, d)
        return output


class DatasetConditionedEncoderDecoder(transformers.PreTrainedModel):
    def __init__(
            self, 
            config,
            dataset_backbone: transformers.PreTrainedModel,
            biencoder: BiEncoder,
        ):
        super().__init__(config=config)
        self.backbone = dataset_backbone
        self.embedding_dim = dataset_backbone.config.hidden_size
        self.hidden_size = self.backbone.config.hidden_size

        # Disable positional embeddings
        num_pos_emb_disabled_modules = 0
        for module in self.backbone.encoder.modules():
            if hasattr(module, "has_relative_attention_bias"):
                if module.has_relative_attention_bias:
                    module.has_relative_attention_bias = False
                    del module.relative_attention_bias
                    module.has_relative_attention_bias = None
                    num_pos_emb_disabled_modules += 1
        print0(f"[DatasetConditionedEncoderDecoder] disabled relative attention in {num_pos_emb_disabled_modules} modules")
        
        num_causal_mask_disabled_modules = 0
        # TODO: Disable causal masking...
        # for module in self.backbone.decoder.modules():
        #     if 'SelfAttention' in module.__class__.__name__:
        #         # disable causal mask
        #         # see code:
        #         # github.com/huggingface/transformers
        #         #       /blob/main/src/transformers
        #         #       /models/t5/modeling_t5.py#440
        #         if module.SelfAttention.is_decoder:
        #             module.SelfAttention.is_decoder = False
        #             num_causal_mask_disabled_modules += 1
        print0(f"[DatasetConditionedEncoderDecoder] disabled causal mask in {num_causal_mask_disabled_modules} modules")

        # Remove shared embedding params. We don't use these and they will mess up DDP param
        # reduction by just sitting around unused.
        del self.backbone.shared
        self.backbone.shared = None
        del self.backbone.encoder.embed_tokens
        self.backbone.encoder.embed_tokens = None

        # TODO: Verify causal masking is properly disabled.
        self.backbone.config.use_cache = False
        self.backbone.decoder.is_decoder = False
        self.backbone.decoder.config.use_cache = False

        # Project biencoder word embeddings
        # self.word_embeddings = word_embeddings
        vocab_bottleneck_dim = 768
        vocab_size = self.backbone.decoder.embed_tokens.weight.shape[0]
        # self.word_embeddings_projection = torch.nn.Sequential(
        #     torch.nn.Linear(self.embedding_dim, vocab_bottleneck_dim),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(vocab_bottleneck_dim, vocab_size)
        # )
        self.output_projection = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, self.config.embedding_output_dim or self.hidden_size)
        )
        self.randomize_dataset_sequence_order = False
        # self.biencoder = biencoder
    
    def forward(
            self, 
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            dataset_embeddings: torch.Tensor
        ) -> torch.Tensor:
        dataset_embeddings = dataset_embeddings[None, :, :] # (b, d) -> (1, b, d)
        dataset_embeddings = dataset_embeddings.to(input_ids).device

        if dataset_embeddings.shape[1] > self.config.transductive_corpus_size:
            # If too many dataset embeddings are passed in, just take the first N until
            # we have the proper number.
            dataset_embeddings = dataset_embeddings[:, :self.config.transductive_corpus_size, :]
        
        batch_size = input_ids.shape[0]
        _, corpus_size, _hidden_dim = dataset_embeddings.shape
        assert _ == 1
        
        # backbone_max_seq_length = self.backbone.config.max_trained_positions
        # assert batch_size + (2 * self.n_soft_prompt + corpus_size) <= backbone_max_seq_length, "too many hard negatives for backbone model"
        soft_prompt = dataset_embeddings
        encoder_attention_mask = torch.ones(
            soft_prompt.shape[0:2],
            dtype=torch.long,
            device=soft_prompt.device,
        )
        # encoder_output = self.backbone.encoder(
        #     inputs_embeds=soft_prompt,
        #     attention_mask=encoder_attention_mask,
        # )
        # encoder_hidden_states = encoder_output.last_hidden_state

        S = len(input_ids)
        soft_prompt = soft_prompt.expand((S, -1, -1))
        encoder_attention_mask = encoder_attention_mask.expand((S, -1, -1))

        if self.training and self.randomize_dataset_sequence_order:
            randomized_order = torch.randperm(corpus_size, device=soft_prompt.device)
            soft_prompt = soft_prompt.gather(
                1, 
                randomized_order[..., None].expand_as(soft_prompt)
            )

        output = self.backbone(
            inputs_embeds=soft_prompt,               
            attention_mask=encoder_attention_mask,   
            decoder_input_ids=input_ids,             
            decoder_attention_mask=attention_mask,
        ) 
        output_vectors = output.last_hidden_state
        # use last token as output vector
        last_token_idxs = attention_mask.sum(1) - 1
        output_pooled = output_vectors[torch.arange(S), last_token_idxs]
        # output_pooled = mean_pool(output.last_hidden_state, attention_mask)
        # average with original vectors
        # TODO: Argparse for pooling strategy...
        # output_vectors = torch.cat((soft_prompt_pooled, output_pooled), dim=1) # (b, d) + (b, d) -> (b, 2d)
        output = self.output_projection(output_pooled) # (b, d) -> (b, d)
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
            print(f"Limiting layers to {config.limit_layers}")
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
        
    def forward(
            self, 
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            dataset_input_ids: Optional[torch.Tensor],
            dataset_attention_mask: Optional[torch.Tensor],
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
        )


class DatasetTransformerEncoderDecoder(transformers.PreTrainedModel):
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
            print(f"Limiting layers to {config.limit_layers}")
            limit_layers(embedder, config.limit_layers)
            limit_layers(dataset_backbone, config.limit_layers)
        
        biencoder_config = copy.deepcopy(config)
        biencoder_config.embedding_output_dim = None
        self.first_stage_model = BiEncoder(
            config=biencoder_config,
            embedder=embedder
        )
        del dataset_backbone
        dataset_backbone = transformers.AutoModel.from_pretrained("t5-base")
        # dataset_backbone = transformers.AutoModel.from_pretrained("EleutherAI/pile-t5-base")
        # embedder_copied = transformers.AutoModel.from_pretrained("EleutherAI/pile-t5-base").encoder
        embedder_copied = copy.deepcopy(embedder)
        self.second_stage_model = DatasetConditionedEncoderDecoder(
            config=config,
            dataset_backbone=dataset_backbone,
            biencoder=BiEncoder(
                config=copy.deepcopy(config),
                embedder=embedder_copied
            ),
        )
        self.temp = config.logit_scale
        if config.disable_dropout:
            disable_dropout(self)

        # Compile modules separately.
        self.first_stage_model = torch.compile(self.first_stage_model)
        self.second_stage_model = torch.compile(self.second_stage_model)
        
    def forward(
            self, 
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            dataset_input_ids: Optional[torch.Tensor],
            dataset_attention_mask: Optional[torch.Tensor],
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
        )


def get_model_class(name: str):
    if name == 'two_head_mlp':
        return TwoEmbeddersWithMLP
    elif name in ['query_independent_dt', 'transductive']: # first name is the old one
        return DatasetTransformer
    elif name in ['transductive__encoder_decoder']: # first name is the old one
        return DatasetTransformerEncoderDecoder
    elif name == 'biencoder':
        return BiEncoder
    else:
        raise ValueError(f'unknown model cls {name}')
