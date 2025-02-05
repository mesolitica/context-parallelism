from context_parallelism import ring_flex_attn
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
        3, batch_size, nheads, seqlen, d, device=device, dtype=dtype, requires_grad=False
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
    print(out)
    out.sum().backward()
    print(q.grad)

"""
CUDA_VISIBLE_DEVICES=2 torchrun \
--nproc_per_node 4 \
--rdzv-endpoint=localhost:29501 \
test/test_ring_flex.py
"""