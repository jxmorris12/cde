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


class EncoderDecoderWithDatasetEmbedder(transformers.PreTrainedModel):
    embedder: transformers.PreTrainedModel
    dataset_embedder: transformers.PreTrainedModel
    def __init__(
            self, 
            config,
            embedder: transformers.PreTrainedModel, 
            dataset_embedder: transformers.PreTrainedModel, 
            dataset_backbone: transformers.PreTrainedModel,
        ):
        super().__init__(config=config)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained('t5-base')
        del dataset_backbone

        if config.limit_layers:
            print(f"Limiting layers to {config.limit_layers}")
            limit_layers(embedder, config.limit_layers)
            limit_layers(dataset_embedder, config.limit_layers)
        
        self.embedder = embedder
        self.dataset_embedder = dataset_embedder

        if hasattr(dataset_embedder, "encoder"):
            self.dataset_embedder = dataset_embedder.encoder
        else:
            self.dataset_embedder = dataset_embedder
        self.dataset_embedder_hidden_size = self.dataset_embedder.config.hidden_size

        # TODO - consider BART or another encoder-decoder.
        disabled_attention_bias_count = 0
        for M in self.embedder.encoder.modules(): 
            if hasattr(M, "has_relative_attention_bias"):
                setattr(M, "has_relative_attention_bias", False)
                # print("> disabled encoder bias")
                disabled_attention_bias_count += 1
        for M in self.embedder.decoder.modules():
            if isinstance(M, transformers.models.t5.modeling_t5.T5LayerCrossAttention):
                setattr(M, "has_relative_attention_bias", False)
                # print("> disabled decoder bias")
                disabled_attention_bias_count += 1
        print(f"Disabled {disabled_attention_bias_count} attention biases")

        self.hidden_size = self.embedder.config.hidden_size
        
        self.input_mlp = torch.nn.Sequential(
            torch.nn.Linear(self.dataset_embedder_hidden_size, self.hidden_size * 2),
            torch.nn.GELU(),
            torch.nn.Linear(self.hidden_size * 2, self.hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
        )

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, self.hidden_size * 2),
            torch.nn.GELU(),
            torch.nn.Linear(self.hidden_size * 2, self.hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
        )
        if config.disable_dropout:
            disable_dropout(self)
        
        self.temp = config.contrastive_temp
       
        self.dataset_positional_embeddings = torch.nn.Parameter(
            torch.randn((1024, self.hidden_size)), requires_grad=True
        )
    
    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            dataset_input_ids: torch.Tensor,
            dataset_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = input_ids.shape[0]
        corpus_size = dataset_input_ids.shape[0]
        dataset_outputs = self.dataset_embedder(
            input_ids=dataset_input_ids,
            attention_mask=dataset_attention_mask
        ).last_hidden_state
        dataset_embeddings = mean_pool(
            hidden_states=dataset_outputs,
            attention_mask=dataset_attention_mask,
        )
        assert len(dataset_embeddings.shape) == 2 # (b, d)
        dataset_embeddings = self.input_mlp(dataset_embeddings)
        # dataset_embeddings = (
        #     dataset_embeddings + self.dataset_positional_embeddings[:batch_size, :]
        # )
        dataset_embeddings = dataset_embeddings[None, :, :].expand(batch_size, -1, -1)
        _corpus_size = dataset_input_ids.shape[0]

        dataset_backbone_attention_mask = torch.ones(
            dataset_embeddings.shape[0:2], 
            device=dataset_embeddings.device,
            dtype=torch.long
        )
        outputs = self.embedder(
            inputs_embeds=dataset_embeddings,
            attention_mask=dataset_backbone_attention_mask,
            decoder_input_ids=input_ids,
            decoder_attention_mask=attention_mask,
        )
        # select last hidden token
        gather_idxs = attention_mask.cumsum(1).argmax(1)
        batch_idxs = torch.arange(batch_size, device=gather_idxs.device)
        embeddings = outputs.last_hidden_state[batch_idxs, gather_idxs]
        assert embeddings.shape == (batch_size, self.hidden_size)
        # project
        output_embeddings = self.mlp(embeddings)
        assert output_embeddings.shape == (batch_size, self.hidden_size)
        return output_embeddings


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
        self.dataset_backbone = dataset_backbone

        # TODO make this a little nicer. (Not every model has 'embeddings...')
        self.dataset_backbone = dataset_backbone
        self.dataset_backbone.embeddings.word_embeddings.weight.requires_grad = False
        self.dataset_backbone.embeddings.word_embeddings.weight.fill_(0.0)

        self.hidden_size = self.embedder.config.hidden_size
        joint_hidden_size = (self.embedder.config.hidden_size + self.dataset_backbone.config.hidden_size)
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
        emb_dtype = self.dataset_backbone.embeddings.word_embeddings.weight.dtype
        dataset_backbone_input_embeddings = dataset_backbone_input_embeddings.to(emb_dtype)
        assert len(dataset_backbone_input_embeddings.shape) == 2 # (b, d)
        dataset_backbone_input_embeddings = dataset_backbone_input_embeddings[:, None, :]
        dataset_backbone_attention_mask = torch.ones(
            dataset_backbone_input_embeddings.shape[0:2], 
            device=dataset_input_ids.device,
            dtype=torch.long
        )
        dataset_intermediate_embeddings = self.dataset_backbone(
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
        self.embedder = embedder
        self.dataset_backbone = dataset_backbone

        self.hidden_size = self.embedder.config.hidden_size

        # TODO make this a little nicer. (Not every model has 'embeddings...')
        self.backbone = dataset_backbone
        # self.backbone.embeddings.word_embeddings.weight.requires_grad = False
        # self.backbone.embeddings.word_embeddings.weight.fill_(0.0)
        self.backbone.config.rotary_emb_fraction = 0.0
        rotary_disabled = 0
        for module in self.backbone.modules():
            if hasattr(module, "rotary_emb_dim"):
                rotary_disabled += 1
                module.rotary_emb_dim = 0
        print(f"disabled {rotary_disabled} rotary modules")

        self.embedding_dim = self.embedder.config.hidden_size
        self.hidden_size = self.backbone.config.hidden_size

        self.n_sequence = 4
        self.prompt_projection = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_dim, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, self.hidden_size * self.n_sequence)
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
        self.gamma = config.gamma
        if config.disable_dropout:
            disable_dropout(self)

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
        outputs = (
            self.embedder(
                input_ids=dataset_input_ids,
                attention_mask=dataset_attention_mask).last_hidden_state
        )
        dataset_embeddings = mean_pool(outputs, dataset_attention_mask) # (b, s, d) -> (b, d)
        dataset_embeddings = dataset_embeddings[None, :, :] # (b, d) -> (1, b, d)
        
        batch_size = input_ids.shape[0]
        dataset_embeddings = self.corpus_projection(dataset_embeddings) # (1, b, d) -> (1, b, d)
        _, corpus_size, hidden_dim = dataset_embeddings.shape
        assert _ == 1
        
        # TODO: we shouldn't need to apply the below constraint if we property disable backbone
        # model positionality.
        backbone_max_seq_length = self.backbone.config.max_position_embeddings
        assert batch_size + (2 * self.n_sequence + corpus_size) <= backbone_max_seq_length, "too many hard negatives for backbone model"

        soft_prompt = torch.ones((1, self.embedding_dim), device=dataset_embeddings.device, dtype=torch.float32)
        soft_prompt = self.prompt_projection(soft_prompt).reshape((1, self.n_sequence, self.hidden_size))
        soft_prompt = torch.cat((soft_prompt, dataset_embeddings), dim=1)
        soft_prompt = soft_prompt.repeat((len(input_ids), 1, 1)) # -> (b, 4+b, d)
        
        inputs_embeds = self.backbone.embeddings.word_embeddings(input_ids) # (b, s) -> (b, s, d)
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

        # print("forward shapes -- input_ids:", input_ids.shape, "output_vectors", output_vectors.shape, "dataset_input_ids", dataset_input_ids.shape)
        # print("forward dataset_input_ids unique numel --", dataset_input_ids.unique().numel())

        # use only these tokens
        n_soft_prompt_tokens = soft_prompt.shape[1]
        output_vectors = output.last_hidden_state[:, n_soft_prompt_tokens:, :]
        attention_mask = attention_mask[:, n_soft_prompt_tokens:]

        # average with original vectors
        output_vectors = mean_pool(output_vectors, attention_mask)
        return self.output_projection(output_vectors)
    


class DatasetTransformerDeeper(transformers.PreTrainedModel):
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
        self.tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased')

        if config.limit_layers:
            print(f"Limiting layers to {config.limit_layers}")
            limit_layers(embedder, config.limit_layers)
            limit_layers(dataset_backbone, config.limit_layers)
        
        self.dataset_embedder = dataset_embedder
        self.embedder = embedder
        self.dataset_backbone = dataset_backbone

        self.hidden_size = self.embedder.config.hidden_size

        # TODO make this a little nicer. (Not every model has 'embeddings...')
        self.backbone = dataset_backbone
        # self.backbone.embeddings.word_embeddings.weight.requires_grad = False
        # self.backbone.embeddings.word_embeddings.weight.fill_(0.0)
        self.backbone.config.rotary_emb_fraction = 0.0
        rotary_disabled = 0
        for module in self.backbone.modules():
            if hasattr(module, "rotary_emb_dim"):
                rotary_disabled += 1
                module.rotary_emb_dim = 0
        print(f"disabled {rotary_disabled} rotary modules")

        self.embedding_dim = self.embedder.config.hidden_size
        self.hidden_size = self.backbone.config.hidden_size

        self.n_sequence = 4
        self.prompt_projection = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_dim, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, self.hidden_size * self.n_sequence)
        )
        self.backbone_output_projection = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, self.hidden_size)
        )
        self.dataset_embedder_projection = torch.nn.Sequential(
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
        self.gamma = config.gamma
        if config.disable_dropout:
            disable_dropout(self)

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
        embedder_outputs = (
            self.embedder(
                input_ids=dataset_input_ids,
                attention_mask=dataset_attention_mask).last_hidden_state
        )
        dataset_embeddings = mean_pool(embedder_outputs, dataset_attention_mask) # (b, s, d) -> (b, d)
        dataset_embeddings = dataset_embeddings[None, :, :] # (b, d) -> (1, b, d)
        
        batch_size = input_ids.shape[0]
        dataset_embeddings = self.corpus_projection(dataset_embeddings) # (1, b, d) -> (1, b, d)
        _, corpus_size, hidden_dim = dataset_embeddings.shape
        assert _ == 1
        
        # TODO: we shouldn't need to apply the below constraint if we property disable backbone
        # model positionality.
        dataset_embedder_max_seq_length = self.dataset_embedder.config.max_position_embeddings
        assert batch_size + (2 * self.n_sequence + corpus_size) <= dataset_embedder_max_seq_length, "too many hard negatives for dataset-embedding model"

        soft_prompt = torch.ones((1, self.embedding_dim), device=dataset_embeddings.device, dtype=torch.float32)
        soft_prompt = self.prompt_projection(soft_prompt).reshape((1, self.n_sequence, self.hidden_size))
        soft_prompt = torch.cat((soft_prompt, dataset_embeddings), dim=1)
        soft_prompt = soft_prompt.repeat((len(input_ids), 1, 1)) # -> (b, 4+b, d)
        
        inputs_embeds = self.dataset_embedder.embeddings.word_embeddings(input_ids) # (b, s) -> (b, s, d)
        inputs_embeds = torch.cat((soft_prompt, inputs_embeds), dim=1) # (v, 4+b+s, d)

        dataset_embedder_attention_mask = torch.ones(
            soft_prompt.shape[0:2],
            dtype=torch.long,
            device=soft_prompt.device,
        )
        dataset_embedder_attention_mask = torch.cat((dataset_embedder_attention_mask, attention_mask), dim=1)
        output = self.dataset_embedder(
            inputs_embeds=inputs_embeds,
            attention_mask=dataset_embedder_attention_mask,
        ) # (1, 4 + b + s, d)
        # trim soft prompt
        dataset_embedder_output_vectors = output.last_hidden_state

        # use only these tokens
        n_soft_prompt_tokens = soft_prompt.shape[1]
        dataset_embedder_output_vectors = output.last_hidden_state[:, n_soft_prompt_tokens:, :]
        dataset_embedder_output_vectors = self.dataset_embedder_projection(dataset_embedder_output_vectors)
        dataset_embedder_attention_mask = dataset_embedder_attention_mask[:, n_soft_prompt_tokens:]

        # prepare inputs for deeper transformer
        backbone_inputs_embeds = self.backbone.embeddings.word_embeddings(input_ids) # (b, s) -> (b, s, d)
        backbone_inputs_embeds = torch.cat((backbone_inputs_embeds, dataset_embedder_output_vectors), dim=1) # (v, 4+b+s, d)
        backbone_attention_mask = torch.cat(
            (attention_mask, dataset_embedder_attention_mask), dim=1
        )

        ##########################################################################################
        # TODO abstract this insane reordering logic into a helper function :-)
        # reorder inputs to move zeros to the end
        # get indices for gather
        backbone_new_idxs = (backbone_attention_mask.cumsum(1) * backbone_attention_mask) - 1
        # backbone_zero_idxs = (1 - backbone_attention_mask).argmax(1)
        # backbone_new_idxs = torch.where(backbone_new_idxs >= 0, backbone_new_idxs, backbone_zero_idxs[:, None])
        # replace -1s with indices of a zero
        # new_idxs_3d = backbone_new_idxs[..., None].repeat((1, 1, backbone_inputs_embeds.shape[2]))
        # new_inputs_embeds = torch.zeros_like(backbone_inputs_embeds, device=backbone_inputs_embeds.device)
        # new_backbone_attention_mask = torch.zeros_like(backbone_attention_mask, device=backbone_attention_mask.device)
        # breakpoint()
        # new_backbone_attention_mask[backbone_new_idxs] = 1
        # backbone_inputs_embeds = backbone_inputs_embeds.gather(1, new_idxs_3d)
        # old_backbone_attention_mask = backbone_attention_mask.clone()
        # breakpoint()
        # backbone_attention_mask = backbone_attention_mask.gather(1, backbone_new_idxs)
        # assert (
        #     (old_backbone_attention_mask.sum(1) == backbone_attention_mask.sum(1)).all()
        # )
   
        # call transformer
        backbone_output = self.backbone(
            inputs_embeds=backbone_inputs_embeds,
            attention_mask=backbone_attention_mask
        )
        backbone_output_vectors = mean_pool(backbone_output.last_hidden_state, backbone_attention_mask)
        return self.backbone_output_projection(backbone_output_vectors)
    


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
    elif name == 'encoder_decoder_de':
        return EncoderDecoderWithDatasetEmbedder
    elif name == 'query_independent_dt':
        return DatasetTransformer
    elif name == 'query_independent_dt_deeper':
        return DatasetTransformerDeeper
    elif name == 'biencoder':
        return BiEncoder
    else:
        raise ValueError(f'unknown model cls {name}')