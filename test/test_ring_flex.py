from context_parallelism import ring_flex_attn
from context_parallelism.utils import causal_mask
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
import torch
import torch.distributed as dist
import os

if __name__ == "__main__":
    world_size = int(os.environ['LOCAL_WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    dist.init_process_group(backend='nccl')

    batch_size = 1
    seqlen = 128
    nheads = 5
    d = 128

    # poor man test
    device = torch.device('cuda:0')
    dtype = torch.bfloat16

    assert seqlen % world_size == 0
    assert d % 8 == 0

    qkv = torch.randn(
        3, batch_size, nheads, seqlen, d, device=device, dtype=dtype, requires_grad=True,
    )
    dist.broadcast(qkv, src=0)
    local_qkv = qkv.chunk(world_size, dim=-2)[local_rank].detach().clone()
    local_qkv.requires_grad = True
    q = local_qkv[0]
    k = local_qkv[1]
    v = local_qkv[2]

    q.retain_grad()
    k.retain_grad()
    v.retain_grad()

    out, lse = ring_flex_attn(q=q, k=k, v=v)
    out_clone = out.clone()
    out.sum().backward()

    q_grad = q.grad.clone()
    k_grad = k.grad.clone()
    v_grad = v.grad.clone()

    print(q_grad.shape, k_grad.shape, v_grad.shape)

    q = qkv[0]
    k = qkv[1]
    v = qkv[2]

    q.retain_grad()
    k.retain_grad()
    v.retain_grad()

    scale = q.shape[-1] ** (-0.5)
    block_mask = create_block_mask(causal_mask, None, None, q.shape[-2], q.shape[-2])
    out = flex_attention(q, k, v, block_mask=block_mask, scale=scale)
    out.sum().backward()

    print((out_clone - out).abs().max())

    print(q.grad.shape)
    print((q_grad - q.grad).abs().max())





"""
CUDA_VISIBLE_DEVICES=2 torchrun \
--nproc_per_node 4 \
--rdzv-endpoint=localhost:29501 \
test/test_ring_flex.py
"""