import torch
import torch.distributed as dist
import math
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from .utils import merge_attention, RingComm, causal_mask, is_compiled_module
from .backward import attention_backward

"""
Quick intro about Blockwise Ring Attention,

For simplicity sake, let say, we have 2 GPUs with seqlen of 4, partitioned to 2, 4 / 2 = 2, 2

For GPU 0, should calculate q0k0v0 + q0k1v1
For GPU 1, should calculate q1k0v0 + q1k1v1
(+) denoted as blockwise attention.

so the attention is like,

      k0  | k1
q0    o x | x x
      o o | x x
      ---------
q1    o o | o x
      o o | o o

For GPU 0

    cp_rank = 0
    cp_world_size = 2
    send = (cp_rank + 1) % cp_world_size = 1
    receive = (cp_rank - 1) % cp_world_size = 1
    k = k0
    v = v0

    step == 0
        if step + 1 != comm.world_size: True
            send k, v to send
            receive v1, v1 from receive

        if not is_causal or step <= comm.rank: True
            calculate attention with is_causal and step == 0, so this causal

        if step + 1 != comm.world_size: True
        k = k1
        v = v1

    step == 1

        if step + 1 != comm.world_size: False
        if not is_causal or step <= comm.rank: False
        if step + 1 != comm.world_size: False

For GPU 1

    cp_rank = 1
    cp_world_size = 2
    send = (cp_rank + 1) % cp_world_size = 0
    receive = (cp_rank - 1) % cp_world_size = 0
    k = k1
    v = v1

    step == 0
        if step + 1 != comm.world_size: True
            send k, v to send
            receive k0, v0 from receive

        if not is_causal or step <= comm.rank: True
            calculate attention with is_causal and step == 0, so this causal

        if step + 1 != comm.world_size: True
        k = k0
        v = v0

    step == 1
        if step + 1 != comm.world_size: False

        if not is_causal or step <= comm.rank: True
            calculate attention with is_causal and step == 1, full attention

        if step + 1 != comm.world_size: False

Everything hit as it is.
"""

def _forward(
    process_group,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    causal: bool = True,
    _compile: bool = False,
):
    """
    q: [B, H, L, D]
    k: [B, H, L, D]
    v: [B, H, L, D]
    """
    comm = RingComm(process_group)

    out = None
    lse = None
    next_k, next_v = None, None

    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            next_k, next_v = comm.send_recv_kv(k, v)

        if not causal or step <= comm.rank:

            with torch.no_grad():
                if causal and step == 0:
                    block_mask = create_block_mask(
                        causal_mask, 
                        None, 
                        None, 
                        q.shape[-2], 
                        q.shape[-2], 
                        device = comm.rank,
                        _compile=_compile
                    )
                else:
                    block_mask = None

                block_out, block_lse = flex_attention(
                    q, 
                    k, 
                    v, 
                    block_mask=block_mask, 
                    scale=scale, 
                    return_lse=True
                )
                out, lse = merge_attention(out, lse, block_out, block_lse)
                out = out.to(q.dtype)

        if step + 1 != comm.world_size:
            comm.wait()
            k, v = next_k, next_v

    return out, lse

def _backward(
    process_group,
    dout: torch.Tensor,
    dlse: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    scale: float,
    causal: bool = True,
    _compile: bool = False,
):
    kv_comm = RingComm(process_group)
    d_kv_comm = RingComm(process_group)
    dq, dk, dv = None, None, None

    next_dk, next_dv = None, None
    next_k, next_v = None, None

    L = q.shape[-2]

    for step in range(kv_comm.world_size):
        if step + 1 != kv_comm.world_size:
            next_k, next_v = kv_comm.send_recv_kv(k, v)

        if not causal or step <= kv_comm.rank:
            
            o = attention_backward(
                query=q,
                key=k,
                value=v,
                out=out,
                logsumexp=lse / math.log(2),
                grad_out=dout,
                grad_logsumexp=dlse,
                scale=scale,
                causal=causal and step == 0
            )
            block_dq_buffer = o[0]
            block_dk_buffer = o[1]
            block_dv_buffer = o[2]
            
            if dq is None:
                dq = block_dq_buffer.to(torch.float32)
                dk = block_dk_buffer.to(torch.float32)
                dv = block_dv_buffer.to(torch.float32)
            else:
                dq += block_dq_buffer
                d_kv_comm.wait()
                dk = block_dk_buffer + next_dk
                dv = block_dv_buffer + next_dv
        
        elif step != 0:
            d_kv_comm.wait()
            dk, dv = next_dk, next_dv

        if step + 1 != kv_comm.world_size:
            kv_comm.wait()
            k, v = next_k, next_v
        
        next_dk, next_dv = d_kv_comm.send_recv_kv(dk, dv)
    
    d_kv_comm.wait()
    
    return dq.to(q.dtype), next_dk.to(q.dtype), next_dv.to(q.dtype)


class RingFlexAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        scale,
        causal,
        group,
        _compile,
    ):
        if scale is None:
            scale = q.shape[-1] ** (-0.5)

        k = k.contiguous()
        v = v.contiguous()
        out, lse = _forward(group, q, k, v, scale=scale, _compile=_compile)
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.scale = scale
        ctx.causal = causal
        ctx.group = group
        ctx._compile = _compile
        return out, lse

    @staticmethod
    def backward(ctx, dout, dlse, *args):
        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = _backward(
            ctx.group,
            dout,
            dlse,
            q,
            k,
            v,
            out,
            lse,
            scale=ctx.scale,
            causal=ctx.causal,
            _compile=ctx._compile,
        )
        return dq, dk, dv, None, None, None, None

def ring_flex_attn(
    q,
    k,
    v,
    scale=None,
    causal=False,
    group=None,
    _compile=False,
):
    global flex_attention, merge_attention

    if _compile and not is_compiled_module(flex_attention):
        flex_attention = torch.compile(flex_attention)
    if _compile and not is_compiled_module(merge_attention):
        merge_attention = torch.compile(merge_attention)
        
    return RingFlexAttnFunc.apply(q, k, v, scale, causal, group, _compile)