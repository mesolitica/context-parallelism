import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process

"""
Ring Attention just different communication from Blockwise attention,
only make sense for distributed message passing.

```bash
torchrun --nproc_per_node=5 ring-attention-string.py
```

Output,

```
0 ['KV_4', 'KV_3', 'KV_2', 'KV_1', 'KV_0']
4 ['KV_3', 'KV_2', 'KV_1', 'KV_0', 'KV_4']
3 ['KV_2', 'KV_1', 'KV_0', 'KV_4', 'KV_3']
1 ['KV_0', 'KV_4', 'KV_3', 'KV_2', 'KV_1'] 
2 ['KV_1', 'KV_0', 'KV_4', 'KV_3', 'KV_2']
```
"""

def next_rank(rank, total_ranks):
    return (rank + 1) % total_ranks

def prev_rank(rank, total_ranks):
    return (rank - 1) % total_ranks

def main():
    world_size = int(os.environ['LOCAL_WORLD_SIZE'])
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend='gloo')

    my_kv_block = f"KV_{local_rank}"
    
    received_kv_blocks = []
    
    send_to = next_rank(local_rank, world_size)
    receive_from = prev_rank(local_rank, world_size)

    current_kv = my_kv_block
    
    for step in range(world_size):
        send_tensor = torch.tensor([ord(c) for c in current_kv], dtype=torch.int)
        recv_tensor = torch.zeros_like(send_tensor)
        
        send_req = dist.isend(send_tensor, dst=send_to)
        recv_req = dist.irecv(recv_tensor, src=receive_from)
        
        send_req.wait()
        recv_req.wait()
        
        received_kv = ''.join([chr(i) for i in recv_tensor.tolist()])
        received_kv_blocks.append(received_kv)
        current_kv = received_kv

    print(local_rank, received_kv_blocks)

if __name__ == "__main__":
    main()