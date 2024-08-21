import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from functools import partial

from spider.lib.nomic_bert.embedding import BertEmbeddings, DynamicNTKRotaryEmbedding, VarLengthRotaryEmbedding

from flash_attn.ops.fused_dense import FusedDense
from flash_attn.modules.mha import FlashCrossAttention

class FlashMHA(nn.Module):
    """Multi-head self-attention and cross-attention"""

    def __init__(
        self,
        config,
    ) -> None:
        """
        num_heads_kv: can be used to toggle MQA / GQA. If None, use num_heads.
        return_residual: whether to return the input x along with the output. This is for
            performance reason: for post-norm architecture, returning the input allows us
            to fuse the backward of nn.Linear with the residual connection.
        """
        super().__init__()
        self.embed_dim = config.n_embd
        self.cross_attn = True
        self.causal = config.causal
        self.use_flash_attn = True
        self.num_heads = config.n_head
        self.num_heads_kv = config.num_heads_kv if getattr(config, "num_heads_kv", None) is not None else self.num_heads
        assert (
            self.num_heads % self.num_heads_kv == 0
        ), "num_heads must be divisible by num_heads_kv"
        assert self.embed_dim % self.num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = self.embed_dim // self.num_heads
        self.rotary_emb_dim = self.head_dim * config.rotary_emb_fraction
        kv_dim = 2 * self.head_dim * self.num_heads_kv

        self.rotary_emb_dim = self.head_dim * config.rotary_emb_fraction
        if self.rotary_emb_dim > 0:
            self.rotary_emb = VarLengthRotaryEmbedding(
                dim=self.rotary_emb_dim,
                base=config.rotary_emb_base,
                scale_base=config.rotary_emb_scale_base,
                interleaved=config.rotary_emb_interleaved,
            )

        fused_bias_fc = config.fused_bias_fc
        if fused_bias_fc and FusedDense is None:
            raise ImportError("fused_dense is not installed")
        linear_cls = nn.Linear if not fused_bias_fc else FusedDense
        linear_resid_cls = partial(FusedDense, return_residual=True)
        wqkv_cls = linear_cls

        self.Wq = linear_cls(self.embed_dim, self.embed_dim, bias=config.qkv_proj_bias)
        self.Wkv = wqkv_cls(self.embed_dim, kv_dim, bias=config.qkv_proj_bias)
        self.register_buffer(
            "norm_factor",
            torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)).to(torch.get_default_dtype()),
            persistent=False,
        )
        self.inner_cross_attn = FlashCrossAttention(
            causal=config.causal, 
            softmax_scale=1.0 / self.norm_factor, 
            attention_dropout=config.embd_pdrop
        )
        self.out_proj = linear_cls(self.embed_dim, self.embed_dim, bias=config.qkv_proj_bias)

    def forward(
        self,
        x,
        x_kv=None,
        key_padding_mask=None,
        cu_seqlens=None,
        cu_seqlens_k=None,
        max_seqlen=None,
        max_seqlen_k=None,
        mixer_subset=None,
        inference_params=None,
        **kwargs,
    ):
        """
        Arguments:
            x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim) if
                cu_seqlens is None and max_seqlen is None, else (total, hidden_dim) where total
                is the is the sum of the sequence lengths in the batch.
            x_kv: (batch, seqlen, hidden_dim), only applicable for cross-attention. If None, use x.
            cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
                of the sequences in the batch, used to index into x. Only applicable when using
                FlashAttention.
            max_seqlen: int. Maximum sequence length in the batch.
            key_padding_mask: boolean mask, True means to keep, False means to mask out.
                (batch, seqlen). Only applicable when not using FlashAttention.
            mixer_subset: for cross-attention only. If not None, will take a subset of x
                before applying the query projection. Useful for e.g., ViT where we only care
                about the CLS token in the last layer.
            inference_params: for generation. Adapted from Megatron-LM (and Apex)
            https://github.com/NVIDIA/apex/blob/3ff1a10f72ec07067c4e44759442329804ac5162/apex/transformer/testing/standalone_transformer_lm.py#L470
        """
        kwargs = (
            {"cu_seqlens": cu_seqlens, "max_seqlen": max_seqlen, **kwargs}
        )
        batch, seqlen = x.shape[:2]
          
        q = self.Wq(x if mixer_subset is None else x[:, mixer_subset])
        kv = self.Wkv(x_kv if x_kv is not None else x)
    
        q = rearrange(q, "... (h d) -> ... h d", d=self.head_dim)
        kv = rearrange(kv, "... (two hkv d) -> ... two hkv d", two=2, d=self.head_dim)
    
        if self.rotary_emb_dim > 0:
            q, kv = self.rotary_emb(
                qkv=q, 
                kv=kv, 
                seqlen_offset=0, 
                cu_seqlens=cu_seqlens, 
                max_seqlen=max_seqlen,
            )
        
        context = self.inner_cross_attn(
            q=q, 
            kv=kv, 
            cu_seqlens=cu_seqlens, 
            cu_seqlens_k=cu_seqlens_k, 
            max_seqlen=max_seqlen,
            max_seqlen_k=max_seqlen_k
        )
        out = self.out_proj(rearrange(context, "... h d -> ... (h d)"))
        return out