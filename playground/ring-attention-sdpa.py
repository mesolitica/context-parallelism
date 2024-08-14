import torch
import torch.nn.functional as F
from xformers.ops.fmha import (
    memory_efficient_attention_forward,
    memory_efficient_attention_backward, 
    memory_efficient_attention_partial,
    merge_attentions
)
import torch
import torch.distributed as dist
import os

"""
Ring Attention just different communication from Blockwise attention,
only make sense for distributed message passing.

To verify that each hosts received all KV blocks, check out ring-attention-string.py

```bash
torchrun --nproc_per_node=5 ring-attention-sdpa.py
```

Output,

```
0 torch.Size([1, 16, 20, 128])
4 torch.Size([1, 16, 20, 128])
1 torch.Size([1, 16, 20, 128])
2 torch.Size([1, 16, 20, 128])
3 torch.Size([1, 16, 20, 128])
```

To verify the integrity of Blockwise attention, check out prefill-sdpa.ipynb
"""

def next_rank(rank, total_ranks):
    return (rank + 1) % total_ranks

def prev_rank(rank, total_ranks):
    return (rank - 1) % total_ranks

def main():
    world_size = int(os.environ['LOCAL_WORLD_SIZE'])
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend='gloo')

    batch_size = 1
    head_num = 16
    dim = 128
    seq_len = 100 // world_size

    q = torch.randn(batch_size, seq_len, head_num, dim).cuda().to(torch.bfloat16)
    k = torch.randn(batch_size, seq_len, head_num, dim).to(torch.bfloat16)
    v = torch.randn(batch_size, seq_len, head_num, dim).to(torch.bfloat16)
    
    send_to = next_rank(local_rank, world_size)
    receive_from = prev_rank(local_rank, world_size)

    outs, max_lse = None, None
    new_denominator = None
    attn_output = None
    new_lse_full = None
    
    for step in range(world_size):

        recv_k = torch.zeros_like(k)
        send_req = dist.isend(k, dst=send_to)
        recv_req = dist.irecv(recv_k, src=receive_from)
        send_req.wait()
        recv_req.wait()

        recv_v = torch.zeros_like(v)
        send_req = dist.isend(v, dst=send_to)
        recv_req = dist.irecv(recv_v, src=receive_from)
        send_req.wait()
        recv_req.wait()

        out_, lse_ = memory_efficient_attention_partial(q, recv_k.cuda(), recv_v.cuda())
        lse_ = lse_.transpose(1, 2)
        out_ = out_.cpu()
        lse_ = lse_.cpu()

        if max_lse is None:
            max_lse = lse_
            adjust_factors = torch.ones_like(lse_).unsqueeze(-1)
            new_denominator = adjust_factors
            attn_output = out_ * adjust_factors
            new_lse_full = lse_
        
        else:
            new_max_lse = torch.maximum(max_lse, lse_)
            
            old_adjust_factors = torch.exp(max_lse - new_max_lse).unsqueeze(-1)
            new_adjust_factors = torch.exp(lse_ - new_max_lse).unsqueeze(-1)
            
            new_denominator = old_adjust_factors * new_denominator + new_adjust_factors
            attn_output = old_adjust_factors * attn_output + new_adjust_factors * out_
            new_lse_full = new_max_lse + torch.log(torch.exp(new_lse_full - new_max_lse) + torch.exp(lse_ - new_max_lse))
            
            max_lse = new_max_lse
    
    attn_output = attn_output / new_denominator
    attn_output = attn_output.transpose(1, 2)
    print(local_rank, attn_output.shape)

if __name__ == "__main__":
    main()