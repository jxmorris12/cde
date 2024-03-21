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
        ):
        super().__init__(config=config)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased')

        if config.limit_layers:
            print(f"Limiting layers to {config.limit_layers}")
            limit_layers(embedder, config.limit_layers)
            limit_layers(dataset_embedder, config.limit_layers)
        
        self.embedder = embedder
        self.dataset_embedder = dataset_embedder

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
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, self.hidden_size * 2),
            torch.nn.GELU(),
            torch.nn.Linear(self.hidden_size * 2, self.hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
        )
        if config.disable_dropout:
            disable_dropout(self)
        
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
        # print("shapes:", input_ids.shape, dataset_input_ids.shape, "(", attention_mask.shape, dataset_attention_mask.shape, ")")
        batch_size = dataset_input_ids.shape[0]
        dataset_outputs = self.dataset_embedder(
            input_ids=dataset_input_ids,
            attention_mask=dataset_attention_mask
        ).last_hidden_state
        dataset_embeddings = mean_pool(
            hidden_states=dataset_outputs,
            attention_mask=dataset_attention_mask,
        )
        assert len(dataset_embeddings.shape) == 2 # (b, d)
        dataset_embeddings = (
            dataset_embeddings + self.dataset_positional_embeddings[:batch_size, :]
        )
        dataset_embeddings = dataset_embeddings[None, :, :].expand(batch_size, -1, -1)
        batch_size = dataset_input_ids.shape[0]
        outputs = self.embedder(
            inputs_embeds=dataset_embeddings,
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
        self.tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased')

        if config.limit_layers:
            print(f"Limiting layers to {config.limit_layers}")
            limit_layers(embedder, config.limit_layers)
            limit_layers(dataset_embedder, config.limit_layers)
            limit_layers(dataset_backbone, config.limit_layers)
        self.embedder = embedder
        self.dataset_embedder = dataset_embedder

        # TODO make this a little nicer. (Not every model has 'embeddings...')
        self.dataset_backbone = dataset_backbone
        self.dataset_backbone.embeddings.position_embeddings.weight.requires_grad = False
        self.dataset_backbone.embeddings.position_embeddings.weight.fill_(0.0)

        # TODO - fix. consider BART or another encoder-decoder.
        # self.backbone = transformers.AutoModel.from_pretrained("t5-small")
        # for block in self.backbone.decoder.block: 
        #     block.layer[0].SelfAttention.has_relative_attention_bias = False

        self.hidden_size = self.embedder.config.hidden_size
        joint_hidden_size = (self.embedder.config.hidden_size + self.dataset_backbone.config.hidden_size)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(joint_hidden_size, joint_hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(joint_hidden_size, self.hidden_size * 2),
            torch.nn.GELU(),
            torch.nn.Linear(self.hidden_size * 2, self.hidden_size),
        )

        if config.disable_dropout:
            disable_dropout(self)
    
    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            dataset_input_ids: torch.Tensor,
            dataset_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        dataset_input_embeddings = mean_pool(
            hidden_states=self.dataset_embedder(
                input_ids=dataset_input_ids,
                attention_mask=dataset_attention_mask
            ).last_hidden_state,
            attention_mask=dataset_attention_mask,
        )
        assert len(dataset_input_embeddings.shape) == 2 # (b, d)
        dataset_input_embeddings = dataset_input_embeddings[:, None, :]
        dataset_intermediate_embeddings = self.dataset_backbone(
            inputs_embeds=dataset_input_embeddings
        ).last_hidden_state
        dataset_embedding = dataset_intermediate_embeddings[:, 0, :].mean(dim=0, keepdim=True)
        batch_size = dataset_input_ids.shape[0]
        with torch.no_grad():
            embeddings = mean_pool(
                hidden_states=self.embedder(
                    input_ids=input_ids,
                    attention_mask=attention_mask).last_hidden_state,
                attention_mask=attention_mask,
            )
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
        self.backbone.embeddings.position_embeddings.weight.requires_grad = False
        self.backbone.embeddings.position_embeddings.weight.fill_(0.0)

        self.embedding_dim = self.embedder.config.hidden_size
        self.hidden_size = self.backbone.config.hidden_size

        self.n_sequence = 4
        self.prompt_projection = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_dim, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, self.hidden_size * self.n_sequence)
        )
        self.query_projection = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_dim, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, self.hidden_size * self.n_sequence)
        )
        self.corpus_projection = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_dim, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, self.hidden_size)
        )

        # whether to share hard negatives between queries.
        # TODO argparse for this.
        self.share_hard_negatives = False

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
        del dataset_input_ids
        del dataset_attention_mask

        outputs = (
            self.embedder(
                input_ids=input_ids,
                attention_mask=attention_mask).last_hidden_state
        )
        document_embeddings = mean_pool(outputs, attention_mask)
        document_embeddings = document_embeddings[None, :, :]
        
        batch_size = input_ids.shape[0]
        document_embeddings = self.corpus_projection(document_embeddings)
        _, corpus_size, hidden_dim = document_embeddings.shape
        assert _ == 1
        
        # TODO: we shouldn't need to apply the below constraint if we property disable backbone
        # model positionality.
        backbone_max_seq_length = self.backbone.config.max_position_embeddings
        assert batch_size + (2 * self.n_sequence + corpus_size) <= backbone_max_seq_length, "too many hard negatives for backbone model"

        soft_prompt = torch.ones((1, self.embedding_dim), device=document_embeddings.device, dtype=torch.float32)
        soft_prompt = self.prompt_projection(soft_prompt).reshape((1, self.n_sequence, self.hidden_size))
        
        inputs_embeds = document_embeddings
        inputs_embeds = torch.cat((soft_prompt, inputs_embeds), dim=1)
        output = self.backbone(
            inputs_embeds=inputs_embeds,
        )
        # trim soft prompt
        output_vectors = output.last_hidden_state[:, self.n_sequence:, :]
        # average with original vectors
        output_vectors = (document_embeddings * self.gamma) + (output_vectors * (1 - self.gamma))
        # return
        return output_vectors.squeeze(dim=0)


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
    elif name == 'biencoder':
        return BiEncoder
    else:
        raise ValueError(f'unknown model cls {name}')