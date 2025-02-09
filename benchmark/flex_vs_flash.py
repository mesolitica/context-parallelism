import os
import torch
import torch.distributed as dist
from tqdm import tqdm
from context_parallelism import ring_flex_attn
from ring_flash_attn import ring_flash_attn_func

def benchmark(seqlen=16384, num_iter=100):
    begin = torch.cuda.Event(enable_timing=True)
    begin.record()

    world_size = int(os.environ['LOCAL_WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])

    batch_size = 1
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

    with torch.no_grad():
        for i in range(2):
            print(f'warming up flex ring attention, {i}')
            ring_flex_attn(q=q, k=k, v=v, causal=True, _compile=True)
    
    begin = torch.cuda.Event(enable_timing=True)
    begin.record()
    
    with torch.no_grad():
        for _ in tqdm(range(num_iter)):
            ring_flex_attn(q=q, k=k, v=v, causal=True, _compile=True)
    
    end = torch.cuda.Event(enable_timing=True)
    end.record()
    torch.cuda.synchronize(device=device)
    time = begin.elapsed_time(end) / 1000.0

    if local_rank == 0:
        print(f"flex ring attention, {seqlen} seqlen {num_iter / time:.3f} iter/s, {time:.3f} sec")

    torch.cuda.empty_cache()

    with torch.no_grad():
        for i in range(2):
            print(f'warming up flash ring attention, {i}')
            ring_flash_attn_func(q=q, k=k, v=v, causal=True)
    
    begin = torch.cuda.Event(enable_timing=True)
    begin.record()
    
    with torch.no_grad():
        for _ in tqdm(range(num_iter)):
            ring_flash_attn_func(q=q, k=k, v=v, causal=True)
    
    end = torch.cuda.Event(enable_timing=True)
    end.record()
    torch.cuda.synchronize(device=device)
    time = begin.elapsed_time(end) / 1000.0

    if local_rank == 0:
        print(f"flash ring attention, {seqlen} seqlen {num_iter / time:.3f} iter/s, {time:.3f} sec")

if __name__ == "__main__":
    dist.init_process_group("nccl")
    benchmark()

"""
CUDA_VISIBLE_DEVICES=2 torchrun \
--nproc_per_node 1 \
--rdzv-endpoint=localhost:29501 \
benchmark/flex_vs_flash.py
"""