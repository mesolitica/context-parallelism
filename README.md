# context-parallelism-xformers

Context Parallelism using Xformers, support,

1. Blockwise Attention, https://arxiv.org/abs/2305.19370
2. Ring Attention, https://arxiv.org/abs/2310.01889
3. Tree Attention, https://arxiv.org/abs/2408.04093

## Why Xformers

Xformers implemented partial attention, https://facebookresearch.github.io/xformers/_modules/xformers/ops/fmha.html#memory_efficient_attention_partial and the parameters are straight forward.