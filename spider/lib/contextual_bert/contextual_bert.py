# Copyright (c) 2022, Tri Dao.
# This BERT implementation is based on our MLPerf 2.0 and MLPerf 2.1 BERT implementation.
# https://github.com/mlcommons/training_results_v2.0/blob/main/HazyResearch/benchmarks/bert/implementations/pytorch/modeling.py
# https://github.com/mlcommons/training_results_v2.1/blob/main/Azure-HazyResearch/benchmarks/bert/implementations/ND96amsr_A100_v4/modeling.py
from typing import Tuple
# Inspired by https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py

import logging
import os
from functools import partial
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
from flash_attn.ops.rms_norm import RMSNorm, dropout_add_rms_norm, rms_norm
from safetensors.torch import load_file as safe_load_file
from transformers import GPT2Config, PreTrainedModel
from transformers.models.bert.modeling_bert import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    BertForPreTrainingOutput,
    SequenceClassifierOutput,
)

from .attention import FlashMHA
from .configuration import ContextualBertConfig

from spider.lib.nomic_bert.attention import FlashAttention
from spider.lib.nomic_bert.block import MLP, GatedMLP
from spider.lib.nomic_bert.embedding import BertEmbeddings
from spider.lib.nomic_bert.bert import remap_bert_state_dict
from spider.lib.nomic_bert.model_utils import filter_shapes, state_dict_from_pretrained

try:
    from flash_attn.ops.fused_dense import FusedDense
except ImportError:
    FusedDense = None

try:
    from flash_attn.ops.layer_norm import dropout_add_layer_norm, layer_norm
except ImportError:
    dropout_add_layer_norm, layer_norm = None, None

try:
    from flash_attn.losses.cross_entropy import CrossEntropyLoss
except ImportError:
    CrossEntropyLoss = None


logger = logging.getLogger(__name__)



class ContextualBlock(nn.Module):
    def __init__(
        self,
        config,
        drop_path_rate1=0.0,
        drop_path_rate2=0.0,
    ):
        """
        For prenorm=True, this Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA -> Dropout -> Add -> LN -> MLP -> Dropout -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Dropout -> Add -> LN -> MHA -> Dropout -> Add -> LN -> MLP, returning both
        the hidden_states (output of the MLP) and the residual.
        This is for performance reasons, as we can fuse the dropout, add and LayerNorm.
        The residual needs to be provided (except for the very first block).

        For prenorm=False, this Block has the same structure as a regular postnorm Transformer
        block: MHA -> Dropout -> Add -> LN -> MLP -> Dropout -> Add -> LN.
        See more: https://github.com/Dao-AILab/flash-attention/issues/216#issuecomment-1546638138
        """
        super().__init__()
        self.prenorm = False
        self.fused_dropout_add_ln = config.fused_dropout_add_ln

        self.attn = FlashAttention(config)
        self.cross_attn = FlashMHA(config)

        activation = (
            F.sigmoid
            if config.activation_function == "glu"
            else (F.silu if config.activation_function == "swiglu" else F.gelu)
        )
        if config.activation_function in ["glu", "swiglu", "geglu"]:
            self.mlp = GatedMLP(
                config.n_embd,
                hidden_features=config.n_inner,
                bias1=config.mlp_fc1_bias,
                bias2=config.mlp_fc2_bias,
                activation=activation,
                fused_bias_fc=config.fused_bias_fc,
            )
        else:
            self.mlp = MLP(
                config.n_embd,
                hidden_features=config.n_inner,
                bias1=config.mlp_fc1_bias,
                bias2=config.mlp_fc2_bias,
                activation=activation,
                fused_bias_fc=config.fused_bias_fc,
            )

        self.dropout1 = nn.Dropout(config.resid_pdrop)
        norm_cls = partial(
            nn.LayerNorm if not config.use_rms_norm else RMSNorm,
            eps=config.layer_norm_epsilon,
        )
        self.norm1 = norm_cls(config.n_embd)
        self.norm2 = norm_cls(config.n_embd)

        self.cross_attn_norm_1 = norm_cls(config.n_embd)
        self.cross_attn_norm_2 = norm_cls(config.n_embd)

        self.dropout2 = nn.Dropout(config.resid_pdrop)
        self.residual_in_fp32 = getattr(config, "residual_in_fp32", False)

        self.drop_path1 = StochasticDepth(drop_path_rate1, mode="row") if drop_path_rate1 > 0.0 else nn.Identity()
        self.drop_path2 = StochasticDepth(drop_path_rate2, mode="row") if drop_path_rate2 > 0.0 else nn.Identity()

        self.layer_scale = getattr(config, "layer_scale", False)
        if self.layer_scale:
            self.ls1 = nn.Parameter(config.layer_scale_init * torch.ones(config.n_embd))
            self.ls2 = nn.Parameter(config.layer_scale_init * torch.ones(config.n_embd))

    def forward(
        self,
        hidden_states: torch.Tensor,
        hidden_states2: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        is_padded_inputs: Optional[bool] = True,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cu_seqlens: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        max_seq_len: Optional[int] = None,
        max_seqlen_k: Optional[int] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: if postnorm, residual=None, If prenorm, hidden_states = Attn/MLP(LN(residual))
            mixer_subset: for cross-attention only. If not None, will take a subset of x
                before applying the query projection. Useful for e.g., ViT where we only care
                about the CLS token in the last layer.
        """
        fused_add_norm_fn = (
            dropout_add_rms_norm if RMSNorm and isinstance(self.norm1, RMSNorm) else dropout_add_layer_norm
        )
    
        assert residual is None
        attn_outputs = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            is_padded_inputs=is_padded_inputs,
            cu_seqlens=cu_seqlens,
            max_seq_len=max_seq_len,
        )
        if not self.fused_dropout_add_ln:
            hidden_states = self.norm1(
                (self.drop_path1(self.dropout1(attn_outputs)) + hidden_states).to(dtype=self.norm1.weight.dtype)
            )
        else:
            if isinstance(self.drop_path1, nn.Identity) or self.drop_path1.p == 0 or not self.training:
                rowscale1 = None
            else:
                rowscale1 = self.drop_path1(
                    torch.ones(attn_outputs.shape[:-1], device=attn_outputs.device, dtype=attn_outputs.dtype)
                )
            hidden_states = fused_add_norm_fn(
                attn_outputs,
                hidden_states,
                self.norm1.weight,
                self.norm1.bias,
                self.dropout1.p if self.training else 0.0,
                self.norm1.eps,
                rowscale=rowscale1,
                prenorm=False,
            )
        
        # Do cross attention with `encoder_hidden_states`
        cross_attention_outputs = self.cross_attn(
            x=hidden_states, 
            x_kv=encoder_hidden_states,
            cu_seqlens=cu_seqlens,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen=max_seq_len,
            max_seqlen_k=max_seqlen_k,
        )

        hidden_states = (hidden_states + cross_attention_outputs)
        
        # Do MLP
        mlp_out = self.mlp(hidden_states)
        if not self.fused_dropout_add_ln:
            hidden_states = self.norm2(
                (self.drop_path2(self.dropout2(mlp_out)) + hidden_states).to(dtype=self.norm2.weight.dtype)
            )
        else:
            if isinstance(self.drop_path2, nn.Identity) or self.drop_path2.p == 0 or not self.training:
                rowscale2 = None
            else:
                rowscale2 = self.drop_path2(
                    torch.ones(mlp_out.shape[:-1], device=mlp_out.device, dtype=mlp_out.dtype)
                )
            hidden_states = fused_add_norm_fn(
                mlp_out,
                hidden_states,
                self.norm2.weight,
                self.norm2.bias,
                self.dropout2.p if self.training else 0.0,
                self.norm2.eps,
                rowscale=rowscale2,
                prenorm=False,
            )
        return hidden_states, None, None


class ContextualBertPreTrainedModel(PreTrainedModel):
    """An abstract class to handle weights initialization and
    a simple interface for dowloading and loading pretrained models.
    """

    config_class = ContextualBertConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Block"]
    _skip_keys_device_placement = "past_key_values"

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config)
        if not isinstance(config, GPT2Config):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `GPT2Config`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                )
            )
        self.config = config

    @classmethod
    def from_pretrained(cls, model_name, config=None, *inputs, **kwargs):
        """
        Instantiate a ContextualBertPreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.

        Params:
            pretrained_model_name_or_path: either:
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a ContextualBertForPretraining instance
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `model.chkpt` a TensorFlow checkpoint
            *inputs, **kwargs: additional input for the specific ContextualBert class
                (ex: num_labels for ContextualBertForSequenceClassification)
        """
        # Instantiate model.
        if config is None:
            config = cls.config_class.from_pretrained(model_name)
        remove_cls = cls != ContextualBertForPreTraining
        remove_bert_prefix = cls != ContextualBertForPreTraining
        ignore_mismatched_shapes = kwargs.pop("ignore_mismatched_sizes", False)
        num_labels = kwargs.pop("num_labels", None)
        rotary_scaling_factor = kwargs.pop("rotary_scaling_factor", None)
        strict = kwargs.pop("strict", True)
        config.rotary_scaling_factor = rotary_scaling_factor
        if config.n_positions <= 0 and config.rotary_emb_fraction > 0:
            config.n_positions = 2048
        if num_labels:
            config.num_labels = num_labels

        if "add_pooling_layer" in kwargs:
            model = cls(config, *inputs, add_pooling_layer=kwargs.pop("add_pooling_layer"))
        else:
            model = cls(config, *inputs)

        # TODO: fix this
        if os.path.exists(model_name):
            model_path = f"{model_name}/pytorch_model.bin"
            if os.path.exists(model_path):
                state_dict = torch.load(f"{model_name}/pytorch_model.bin")
            else:
                model_path = f"{model_name}/model.safetensors"
                if not os.path.exists(model_path):
                    raise ValueError(f"Model path {model_path} not found")
                state_dict = safe_load_file(model_path)

            if ignore_mismatched_shapes:
                state_dict = filter_shapes(state_dict, model)
            load_return = model.load_state_dict(state_dict, strict=False)
        else:
            # TODO: can probably check config class and see if we need to remap from a bert model
            state_dict = state_dict_from_pretrained(model_name)
            state_dict = remap_bert_state_dict(
                state_dict,
                config,
                remove_bert=remove_bert_prefix,
                remove_cls_weights=remove_cls,
                add_pooling_layer=getattr(config, "add_pooling_layer", False),
            )
            if ignore_mismatched_shapes:
                state_dict = filter_shapes(state_dict, model)

            load_return = model.load_state_dict(state_dict, strict=False)
        # logger.warning(load_return)
        return model

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, ContextualBertEncoder):
            module.gradient_checkpointing = value


# https://github.com/huggingface/transformers/blob/7032e0203262ebb2ebf55da8d2e01f873973e835/src/transformers/models/bert/modeling_bert.py#L748
def _init_weights(module, initializer_range=0.02):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=initializer_range)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)
        if module.padding_idx is not None:
            nn.init.zeros_(module.weight[module.padding_idx])


class ContextualBertEncoder(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.layers = nn.ModuleList([ContextualBlock(config) for _ in range(config.n_layer)])
        self.gradient_checkpointing = False
        self.config = config

    def forward(
        self,
        hidden_states: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        is_padded_inputs: Optional[bool] = True,
    ):
        """If subset_mask is not None, we only want output for the subset of the sequence.
        This means that we only compute the last layer output for these tokens.
        subset_mask: (batch, seqlen), dtype=torch.bool
        """
        hidden_states2 = None
        residual = None

        assert encoder_hidden_states.shape[0] == hidden_states.shape[0]

        batch, seqlen = hidden_states.shape[:2]
        hidden_states, indices, cu_seqlens, max_seq_len = unpad_input(hidden_states, attention_mask)
        encoder_attention_mask = torch.ones(encoder_hidden_states.shape[0:2], dtype=torch.bool, device=encoder_hidden_states.device)
        encoder_hidden_states, encoder_indices, cu_seqlens_k, max_seqlen_k = unpad_input(encoder_hidden_states, encoder_attention_mask)

        for i, layer in enumerate(self.layers):
            hidden_states, hidden_states2, residual = layer(
                hidden_states,
                hidden_states2,
                residual,
                attention_mask,
                position_ids,
                None,
                is_padded_inputs,
                output_attentions,
                use_cache,
                cu_seqlens=cu_seqlens,
                max_seq_len=max_seq_len,
                encoder_hidden_states=encoder_hidden_states,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_k=max_seqlen_k,
            )
        hidden_states = pad_input(hidden_states, indices, batch, seqlen)
        return hidden_states


class ContextualBertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        fused_bias_fc = getattr(config, "fused_bias_fc", False)
        if fused_bias_fc and FusedDense is None:
            raise ImportError("fused_dense is not installed")
        linear_cls = nn.Linear if not fused_bias_fc else FusedDense
        self.dense = linear_cls(config.n_embd, config.n_embd)
        self.activation = nn.Tanh()

    def forward(self, hidden_states, pool=True):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0] if pool else hidden_states
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ContextualBertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        fused_bias_fc = getattr(config, "fused_bias_fc", False)
        if fused_bias_fc and FusedDense is None:
            raise ImportError("fused_dense is not installed")
        self.fused_dropout_add_ln = getattr(config, "fused_dropout_add_ln", False)
        if self.fused_dropout_add_ln and layer_norm is None:
            raise ImportError("dropout_add_layer_norm is not installed")
        linear_cls = nn.Linear if not fused_bias_fc else FusedDense
        self.dense = linear_cls(config.n_embd, config.n_embd, bias=config.mlp_fc1_bias)
        approximate = "tanh" if config.activation_function in ["gelu_new", "gelu_fast", "gelu_pytorch_tanh"] else "none"
        if config.activation_function == "swiglu":
            self.transform_act_fn = F.silu
        else:
            self.transform_act_fn = nn.GELU(approximate=approximate)
        norm_cls = partial(
            nn.LayerNorm if not config.use_rms_norm else RMSNorm,
            eps=config.layer_norm_epsilon,
        )
        self.layer_norm = norm_cls(config.n_embd)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        if not self.fused_dropout_add_ln:
            hidden_states = self.layer_norm(hidden_states)
        else:
            if isinstance(self.layer_norm, RMSNorm):
                hidden_states = rms_norm(hidden_states, self.layer_norm.weight, self.layer_norm.eps)
            elif isinstance(self.layer_norm, nn.LayerNorm):
                hidden_states = layer_norm(
                    hidden_states, self.layer_norm.weight, self.layer_norm.bias, self.layer_norm.eps
                )
            else:
                raise ValueError(f"Unsupported layer norm class: {self.layer_norm.__class__.__name__}")

        return hidden_states


class ContextualBertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        fused_bias_fc = getattr(config, "fused_bias_fc", False)
        if fused_bias_fc and FusedDense is None:
            raise ImportError("fused_dense is not installed")
        linear_cls = nn.Linear if not fused_bias_fc else FusedDense

        self.transform = ContextualBertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        # TODO: assumes that mlp_fc1_bias == mlp_fc2_bias, we should enforce that somewhere
        self.decoder = linear_cls(config.n_embd, config.vocab_size, bias=config.mlp_fc1_bias)

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class ContextualBertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = ContextualBertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class ContextualBertModel(ContextualBertPreTrainedModel):
    def __init__(self, config: GPT2Config, add_pooling_layer=True):
        super().__init__(config)
        self.pad_vocab_size_multiple = getattr(config, "pad_vocab_size_multiple", 1)
        if config.vocab_size % self.pad_vocab_size_multiple != 0:
            config.vocab_size += self.pad_vocab_size_multiple - (config.vocab_size % self.pad_vocab_size_multiple)
        self.fused_dropout_add_ln = getattr(config, "fused_dropout_add_ln", False)
        if self.fused_dropout_add_ln and layer_norm is None:
            raise ImportError("dropout_add_layer_norm is not installed")
        assert config.activation_function in [
            "gelu",
            "gelu_new",
            "gelu_fast",
            "gelu_pytorch_tanh",
            "swiglu",
            "geglu",
            "glu",
        ]

        self.embeddings = BertEmbeddings(config)
        self.emb_drop = nn.Dropout(config.resid_pdrop)
        self.emb_ln = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.encoder = ContextualBertEncoder(config)
        self.pooler = ContextualBertPooler(config) if add_pooling_layer else None

        self.apply(partial(_init_weights, initializer_range=config.initializer_range))

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        token_type_ids=None,
        attention_mask=None,
        masked_tokens_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
    ):
        """If masked_tokens_mask is not None (i.e. last_layer_subset == True in ContextualBertForPreTraining),
        we only want the output for the masked tokens. This means that we only compute the last
        layer output for these tokens.
        masked_tokens_mask: (batch, seqlen), dtype=torch.bool
        """
        assert (inputs_embeds is None) or (input_ids is None)
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
        # TD [2022-12:18]: Don't need to force residual in fp32
        # BERT puts embedding LayerNorm before embedding dropout.
        if not self.fused_dropout_add_ln:
            hidden_states = self.emb_ln(hidden_states)
        else:
            hidden_states = layer_norm(hidden_states, self.emb_ln.weight, self.emb_ln.bias, self.emb_ln.eps)
        hidden_states = self.emb_drop(hidden_states)

        if masked_tokens_mask is not None:
            batch_size, seqlen = input_ids.shape[:2]
            # We also need the first column for the CLS token
            first_col_mask = torch.zeros(batch_size, seqlen, dtype=torch.bool, device=input_ids.device)
            first_col_mask[:, 0] = True
            subset_mask = masked_tokens_mask | first_col_mask
        else:
            subset_mask = None

        sequence_output = self.encoder(
            hidden_states, 
            attention_mask=attention_mask, 
            encoder_hidden_states=encoder_hidden_states
        )

        if masked_tokens_mask is None:
            pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        else:
            # TD [2022-03-01]: the indexing here is very tricky.
            if attention_mask is not None:
                subset_idx = subset_mask[attention_mask]
                pool_input = sequence_output[first_col_mask[attention_mask][subset_idx]]
                sequence_output = sequence_output[masked_tokens_mask[attention_mask][subset_idx]]
            else:
                pool_input = sequence_output[first_col_mask[subset_mask]]
                sequence_output = sequence_output[masked_tokens_mask[subset_mask]]
            pooled_output = self.pooler(pool_input, pool=False) if self.pooler is not None else None

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
        )


class ContextualBertForPreTraining(ContextualBertPreTrainedModel):
    _tied_weights_keys = ["predictions.decoder.bias", "cls.predictions.decoder.weight"]

    def __init__(self, config: GPT2Config):
        super().__init__(config)
        # If dense_seq_output, we only need to pass the hidden states for the masked out tokens
        # (around 15%) to the classifier heads.
        self.dense_seq_output = getattr(config, "dense_seq_output", False)
        # If last_layer_subset, we only need the compute the last layer for a subset of tokens
        # (e.g., the tokens we need to compute the masked LM loss and the next-sentence prediction).
        self.last_layer_subset = getattr(config, "last_layer_subset", False)
        if self.last_layer_subset:
            assert self.dense_seq_output, "last_layer_subset requires dense_seq_output"
        use_xentropy = getattr(config, "use_xentropy", False)
        if use_xentropy and CrossEntropyLoss is None:
            raise ImportError("xentropy_cuda is not installed")
        loss_cls = nn.CrossEntropyLoss if not use_xentropy else partial(CrossEntropyLoss, inplace_backward=True)

        self.bert = ContextualBertModel(config, add_pooling_layer=getattr(config, "add_pooling_layer", False))
        self.cls = ContextualBertPreTrainingHeads(config)
        self.mlm_loss = loss_cls()

        # Initialize weights and apply final processing
        self.apply(partial(_init_weights, initializer_range=config.initializer_range))
        self.tie_weights()

    def tie_weights(self):
        self.cls.predictions.decoder.weight = self.bert.embeddings.word_embeddings.weight

    def forward(
        self,
        input_ids,
        position_ids=None,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
    ):
        """
        If labels are provided, they must be -100 for masked out tokens (as specified in the attention
        mask).
        Outputs:
            if `labels` and `next_sentence_label` are not `None`:
                Outputs the total_loss which is the sum of the masked language modeling loss and the next
                sentence classification loss.
            if `labels` or `next_sentence_label` is `None`:
                Outputs a tuple comprising
                - the masked language modeling logits of shape [batch_size, sequence_length, vocab_size], and
                - the next sentence classification logits of shape [batch_size, 2].

        """
        masked_tokens_mask = labels > 0 if (self.last_layer_subset and labels is not None) else None
        outputs = self.bert(
            input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask.bool() if attention_mask is not None else None,
            masked_tokens_mask=masked_tokens_mask,
        )
        sequence_output, pooled_output = outputs.last_hidden_state, outputs.pooler_output
        if self.dense_seq_output and labels is not None:
            masked_token_idx = torch.nonzero(labels.flatten() >= 0, as_tuple=False).flatten()
            if not self.last_layer_subset:
                sequence_output = index_first_axis(rearrange(sequence_output, "b s d -> (b s) d"), masked_token_idx)
        prediction_scores = self.cls(sequence_output)

        total_loss = None
        if labels is not None:
            if self.dense_seq_output and labels is not None:  # prediction_scores are already flattened
                masked_lm_loss = self.mlm_loss(prediction_scores, labels.flatten()[masked_token_idx])
            else:
                masked_lm_loss = self.mlm_loss(
                    rearrange(prediction_scores, "... v -> (...) v"),
                    rearrange(labels, "... -> (...)"),
                )
            total_loss = masked_lm_loss.float()

        return BertForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
        )

