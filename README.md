# context-parallelism

Context Parallelism, support,

1. Blockwise Attention, https://arxiv.org/abs/2305.19370
2. Ring Attention, https://arxiv.org/abs/2310.01889
3. Tree Attention, https://arxiv.org/abs/2408.04093

## 2024-10-15 update

Drop Xformers because partial attention from Xformers does not support custom masking, and this necessary especially for causal. So we build our flash attention custom masking forked from https://github.com/alexzhang13/flashattention2-custom-mask
