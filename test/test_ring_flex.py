from context_parallelism import ring_flex_attn
from context_parallelism.utils import causal_mask
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
import torch
import torch.distributed as dist
import os
import math

if __name__ == "__main__":
    world_size = int(os.environ['LOCAL_WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    dist.init_process_group(backend='nccl')

    batch_size = 1
    seqlen = 126
    nheads = 5
    d = 128

    device = torch.device(f'cuda:{local_rank}')
    dtype = torch.bfloat16

    assert seqlen % world_size == 0
    assert d % 8 == 0

    qkv = torch.randn(
        3, batch_size, nheads, seqlen, d, device=device, dtype=dtype, requires_grad=True,
    )
    dist.broadcast(qkv, src=0)

    dout = torch.randn(batch_size, nheads, seqlen, d, device=device, dtype=dtype)
    dist.broadcast(dout, src=0)

    local_qkv = qkv.chunk(world_size, dim=-2)[local_rank].detach().clone()
    local_qkv.requires_grad = True

    local_dout = dout.chunk(world_size, dim=2)[local_rank].detach().clone()

    q = local_qkv[0]
    k = local_qkv[1]
    v = local_qkv[2]

    q.retain_grad()
    k.retain_grad()
    v.retain_grad()

    out, lse = ring_flex_attn(q=q, k=k, v=v, causal=True)
    out_clone = out.clone()
    lse_clone = lse.clone()
    out.backward(local_dout)

    q_grad = q.grad.clone()
    k_grad = k.grad.clone()
    v_grad = v.grad.clone()

    q = qkv[0]
    k = qkv[1]
    v = qkv[2]

    q.retain_grad()
    k.retain_grad()
    v.retain_grad()

    scale = q.shape[-1] ** (-0.5)
    block_mask = create_block_mask(causal_mask, None, None, q.shape[-2], q.shape[-2], device = local_rank)
    out, lse = flex_attention(q, k, v, block_mask=block_mask, scale=scale, return_lse = True)
    out.backward(dout)

    length = local_qkv.shape[-2]
    start_length = int(local_rank * length)
    end_length = int((local_rank + 1) * length)

    print(local_rank, 'out', (out_clone - out[:,:,start_length:end_length]).abs().max())
    print(local_rank, 'lse', (lse_clone - lse[:,:,start_length:end_length]).abs().max())
    print(local_rank, 'q', (q_grad - q.grad[:,:,start_length:end_length]).abs().max())
    print(local_rank, 'k', (k_grad - k.grad[:,:,start_length:end_length]).abs().max())
    print(local_rank, 'v', (v_grad - v.grad[:,:,start_length:end_length]).abs().max())

    assert (out_clone - out[:,:,start_length:end_length]).abs().max() < 1e-2
    assert (q_grad - q.grad[:,:,start_length:end_length]).abs().max() < 1e-2
    assert (k_grad - k.grad[:,:,start_length:end_length]).abs().max() < 1e-2
    assert (v_grad - v.grad[:,:,start_length:end_length]).abs().max() < 1e-2


"""
CUDA_VISIBLE_DEVICES=1,2 torchrun \
--nproc_per_node 2 \
--rdzv-endpoint=localhost:29501 \
test/test_ring_flex.py
"""