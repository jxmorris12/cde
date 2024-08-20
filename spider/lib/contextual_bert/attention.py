import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from functools import partial

class FlashCrossAttention(nn.Module):
    """Multi-head self-attention and cross-attention"""

    def __init__(
        self,
        config,
        # embed_dim,
        # num_heads,
        # num_heads_kv=None,
        # cross_attn=False,
        # qkv_proj_bias=True,
        # out_proj_bias=True,
        # dropout=0.0,
        # softmax_scale=None,
        # causal=False,
        # layer_idx=None,
        # dwconv=False,
        # rotary_emb_dim=0,
        # rotary_emb_base=10000.0,
        # rotary_emb_scale_base=None,
        # rotary_emb_interleaved=False,
        # window_size=(-1, -1),
        # fused_bias_fc=False,
        # use_flash_attn=False,
        # return_residual=False,
        # checkpointing=False,
        # device=None,
        # dtype=None,
    ) -> None:
        """
        num_heads_kv: can be used to toggle MQA / GQA. If None, use num_heads.
        return_residual: whether to return the input x along with the output. This is for
            performance reason: for post-norm architecture, returning the input allows us
            to fuse the backward of nn.Linear with the residual connection.
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = config.n_embd
        self.cross_attn = True
        self.causal = causal
        self.layer_idx = layer_idx
        self.dwconv = dwconv
        self.rotary_emb_dim = rotary_emb_dim
        self.use_flash_attn = use_flash_attn
        self.return_residual = return_residual
        self.checkpointing = checkpointing
        if window_size != (-1, -1):
            assert use_flash_attn, "Local (sliding window) attention code path requires flash_attn"

        self.num_heads = num_heads
        self.num_heads_kv = num_heads_kv if num_heads_kv is not None else num_heads
        assert (
            self.num_heads % self.num_heads_kv == 0
        ), "num_heads must be divisible by num_heads_kv"
        assert self.embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        qkv_dim = self.head_dim * (self.num_heads + 2 * self.num_heads_kv)
        kv_dim = 2 * self.head_dim * self.num_heads_kv

        if self.rotary_emb_dim > 0:
            assert not cross_attn, "MHA with rotary embedding does not support cross-attention yet"
            assert RotaryEmbedding is not None, "rotary_emb is not installed"
            self.rotary_emb = RotaryEmbedding(
                self.rotary_emb_dim,
                base=rotary_emb_base,
                scale_base=rotary_emb_scale_base,
                interleaved=rotary_emb_interleaved,
                device=device,
            )

        if fused_bias_fc and FusedDense is None:
            raise ImportError("fused_dense is not installed")
        linear_cls = nn.Linear if not fused_bias_fc else FusedDense
        linear_resid_cls = (
            LinearResidual if not fused_bias_fc else partial(FusedDense, return_residual=True)
        )
        wqkv_cls = linear_cls if not self.return_residual else linear_resid_cls
        inner_attn_cls = (
            partial(FlashSelfAttention, window_size=window_size)
            if use_flash_attn
            else SelfAttention
        )
        inner_cross_attn_cls = (
            partial(FlashCrossAttention, window_size=window_size)
            if use_flash_attn
            else CrossAttention
        )
        if not self.cross_attn:
            self.Wqkv = wqkv_cls(embed_dim, qkv_dim, bias=qkv_proj_bias, **factory_kwargs)
        else:
            self.Wq = linear_cls(embed_dim, embed_dim, bias=qkv_proj_bias, **factory_kwargs)
            self.Wkv = wqkv_cls(embed_dim, kv_dim, bias=qkv_proj_bias, **factory_kwargs)
        if self.dwconv:
            if self.num_heads_kv == self.num_heads:
                self.dwconv_qkv = nn.Conv1d(
                    qkv_dim, qkv_dim, kernel_size=3, padding=2, groups=qkv_dim
                )
            else:
                self.dwconv_q = nn.Conv1d(
                    embed_dim, embed_dim, kernel_size=3, padding=2, groups=embed_dim
                )
                self.dwconv_kv = nn.Conv1d(kv_dim, kv_dim, kernel_size=3, padding=2, groups=kv_dim)
        self.inner_attn = inner_attn_cls(
            causal=causal,
            softmax_scale=softmax_scale,
            attention_dropout=dropout,
        )
        self.inner_cross_attn = inner_cross_attn_cls(
            causal=causal, softmax_scale=softmax_scale, attention_dropout=dropout
        )
        self.out_proj = linear_cls(embed_dim, embed_dim, bias=out_proj_bias, **factory_kwargs)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None):
        dtype = self.out_proj.weight.dtype if dtype is None else dtype
        device = self.out_proj.weight.device
        return torch.empty(
            batch_size,
            max_seqlen,
            2,
            self.num_heads_kv,
            self.head_dim,
            dtype=dtype,
            device=device,
        )

    def _update_kv_cache(self, kv, inference_params):
        """kv: (batch_size, seqlen, 2, nheads, head_dim) or (batch_size, 1, 2, nheads, head_dim)"""
        assert not self.dwconv, "Generation does not support dwconv yet"
        assert self.layer_idx is not None, "Generation requires layer_idx in the constructor"
        return _update_kv_cache(kv, inference_params, self.layer_idx)

    def _apply_rotary_update_kvcache_attention(self, q, kv, inference_params):
        """
        Fast path that combine 3 steps: apply rotary to Q and K, update kv cache, and apply attention.
        q: (batch_size, seqlen_q, nheads, head_dim)
        kv: (batch_size, seqlen_k, 2, nheads_kv, head_dim)
        """
        assert inference_params is not None and inference_params.seqlen_offset > 0
        assert self.use_flash_attn
        if self.rotary_emb_dim > 0:
            assert self.rotary_emb.scale is None, "This code path does not support xPos"
            self.rotary_emb._update_cos_sin_cache(
                inference_params.max_seqlen, device=q.device, dtype=q.dtype
            )
            rotary_cos, rotary_sin = self.rotary_emb._cos_cached, self.rotary_emb._sin_cached
        else:
            rotary_cos, rotary_sin = None, None
        batch = q.shape[0]
        kv_cache = inference_params.key_value_memory_dict[self.layer_idx][:batch]
        cache_seqlens = (
            inference_params.lengths_per_sample[:batch]
            if inference_params.lengths_per_sample is not None
            else inference_params.seqlen_offset
        )
        context = flash_attn_with_kvcache(
            q,
            kv_cache[:, :, 0],
            kv_cache[:, :, 1],
            kv[:, :, 0],
            kv[:, :, 1],
            rotary_cos=rotary_cos,
            rotary_sin=rotary_sin,
            cache_seqlens=cache_seqlens,
            softmax_scale=self.inner_cross_attn.softmax_scale,
            causal=self.inner_cross_attn.causal,
            rotary_interleaved=self.rotary_emb.interleaved if self.rotary_emb_dim > 0 else False,
        )
        return context

    def _update_kvcache_attention(self, q, kv, inference_params):
        """Write kv to inference_params, then do attention"""
        if (
            inference_params.seqlen_offset == 0
            or flash_attn_with_kvcache is None
            or not self.use_flash_attn
        ):
            # TODO: this only uses seqlen_offset and not lengths_per_sample.
            kv = self._update_kv_cache(kv, inference_params)
            return self.inner_cross_attn(q, kv)
        else:
            batch = q.shape[0]
            kv_cache = inference_params.key_value_memory_dict[self.layer_idx][:batch]
            cache_seqlens = (
                inference_params.lengths_per_sample[:batch]
                if inference_params.lengths_per_sample is not None
                else inference_params.seqlen_offset
            )
            return flash_attn_with_kvcache(
                q,
                kv_cache[:, :, 0],
                kv_cache[:, :, 1],
                kv[:, :, 0],
                kv[:, :, 1],
                cache_seqlens=cache_seqlens,
                softmax_scale=self.inner_cross_attn.softmax_scale,
                causal=self.inner_cross_attn.causal,
            )

    def forward(
        self,
        x,
        x_kv=None,
        key_padding_mask=None,
        cu_seqlens=None,
        max_seqlen=None,
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
        if cu_seqlens is not None:
            assert max_seqlen is not None
            assert key_padding_mask is None
            assert self.use_flash_attn
            assert not self.dwconv
            assert self.rotary_emb_dim == 0
        if key_padding_mask is not None:
            assert cu_seqlens is None
            assert max_seqlen is None
            assert not self.use_flash_attn
        if inference_params is not None:
            assert key_padding_mask is None
            assert cu_seqlens is None and max_seqlen is None
            assert not self.dwconv

        kwargs = (
            {"cu_seqlens": cu_seqlens, "max_seqlen": max_seqlen, **kwargs}
            if self.use_flash_attn
            else {"key_padding_mask": key_padding_mask, **kwargs}
        )
        seqlen_offset = (
            0
            if inference_params is None
            else (
                inference_params.lengths_per_sample
                if inference_params.lengths_per_sample is not None
                else inference_params.seqlen_offset
            )
        )
        rotary_max_seqlen = inference_params.max_seqlen if inference_params is not None else None
        batch, seqlen = x.shape[:2]
        if not self.cross_attn and self.num_heads_kv == self.num_heads:
            assert x_kv is None and mixer_subset is None
            if not self.return_residual:
                qkv = self.Wqkv(x)
            else:
                qkv, x = self.Wqkv(x)
            if self.dwconv:
                qkv = rearrange(
                    self.dwconv_qkv(rearrange(qkv, "b s d -> b d s"))[..., :-2], "b d s -> b s d"
                ).contiguous()
            qkv = rearrange(qkv, "... (three h d) -> ... three h d", three=3, d=self.head_dim)
            if (
                inference_params is None
                or inference_params.seqlen_offset == 0
                or (self.rotary_emb_dim == 0 or self.rotary_emb_dim % 16 != 0)
                or not self.use_flash_attn
            ):
                if self.rotary_emb_dim > 0:
                    qkv = self.rotary_emb(
                        qkv, seqlen_offset=seqlen_offset, max_seqlen=rotary_max_seqlen
                    )
                if inference_params is None:
                    if not self.checkpointing:
                        context = self.inner_attn(qkv, **kwargs)
                    else:
                        context = torch.utils.checkpoint.checkpoint(self.inner_attn, qkv, **kwargs)
                else:
                    context = self._update_kvcache_attention(
                        qkv[:, :, 0], qkv[:, :, 1:], inference_params
                    )
            else:
                context = self._apply_rotary_update_kvcache_attention(
                    qkv[:, :, 0], qkv[:, :, 1:], inference_params
                )
        else:
            if self.cross_attn:
                if not self.return_residual:
                    q = self.Wq(x if mixer_subset is None else x[:, mixer_subset])
                    kv = self.Wkv(x_kv if x_kv is not None else x)
                else:
                    if x_kv is not None:
                        kv, x_kv = self.Wkv(x_kv)
                    else:
                        kv, x = self.Wkv(x)
                    q = self.Wq(x if mixer_subset is None else x[:, mixer_subset])
            else:
                assert self.num_heads_kv != self.num_heads
                if not self.return_residual:
                    qkv = self.Wqkv(x)
                else:
                    qkv, x = self.Wqkv(x)
                q = qkv[..., : self.num_heads * self.head_dim]
                kv = qkv[..., self.num_heads * self.head_dim :]
            q = rearrange(q, "... (h d) -> ... h d", d=self.head_dim)
            kv = rearrange(kv, "... (two hkv d) -> ... two hkv d", two=2, d=self.head_dim)
            if self.dwconv:
                q = rearrange(
                    self.dwconv_q(rearrange(q, "b s d -> b d s"))[..., :-2], "b d s -> b s d"
                ).contiguous()
                kv = rearrange(
                    self.dwconv_kv(rearrange(kv, "b s d -> b d s"))[..., :-2], "b d s -> b s d"
                ).contiguous()
            if (
                inference_params is None
                or inference_params.seqlen_offset == 0
                or (self.rotary_emb_dim == 0 or self.rotary_emb_dim % 16 != 0)
                or not self.use_flash_attn
            ):
                if self.rotary_emb_dim > 0:
                    q, kv = self.rotary_emb(
                        q, kv, seqlen_offset=seqlen_offset, max_seqlen=rotary_max_seqlen
                    )
                if inference_params is None:
                    if not self.checkpointing:
                        context = self.inner_cross_attn(q, kv, **kwargs)
                    else:
                        context = torch.utils.checkpoint.checkpoint(
                            self.inner_cross_attn, q, kv, **kwargs
                        )
                else:
                    context = self._update_kvcache_attention(q, kv, inference_params)
            else:
                context = self._apply_rotary_update_kvcache_attention(q, kv, inference_params)
        out = self.out_proj(rearrange(context, "... h d -> ... (h d)"))
        return out if not self.return_residual else (out, x)