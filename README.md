# tti3

### Install

install pytorch w/ cuda, install requirements:
```bash
uv pip install -r requirements.txt
```

then install FlashAttention:
```bash
uv pip install --no-cache-dir flash-attn --no-build-isolation git+https://github.com/HazyResearch/flash-attention.git#subdirectory=csrc/rotary git+https://github.com/HazyResearch/flash-attention.git#subdirectory=csrc/layer_norm git+https://github.com/HazyResearch/flash-attention.git#subdirectory=csrc/fused_dense_lib git+https://github.com/HazyResearch/flash-attention.git#subdirectory=csrc/xentropy
```
make sure ninja is installed first (`uv pip install ninja`) to make flash attention installation ~50x faster.