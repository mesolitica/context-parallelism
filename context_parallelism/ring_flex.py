import torch
import torch.distributed as dist
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from .utils import merge_attention, RingComm, causal_mask

def _forward(
    process_group,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    causal: bool = True,
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
            if causal and step == 0:
                block_mask = create_block_mask(causal_mask, None, None, q.shape[-2], q.shape[-2])
            else:
                block_mask = None

            block_out, block_lse = flex_attention(q, k, v, block_mask=block_mask, scale=scale, return_lse=True)
            out, lse = merge_attention(out, lse, block_out, block_lse)

        if step + 1 != comm.world_size:
            comm.wait()
            k, v = next_k, next_v

    return out, lse

def _backward(
    process_group,
    dout,
    q,
    k,
    v,
    out,
    lse,
    scale,
    causal=True,
):
    kv_comm = RingComm(process_group)
    d_kv_comm = RingComm(process_group)
    dq, dk, dv = None, None, None
    next_dk, next_dv = None, None

    next_dk, next_dv = None, None
    next_k, next_v = None, None

    L = q.shape[-2]

    for step in range(kv_comm.world_size):
        if step + 1 != kv_comm.world_size:
            next_k, next_v = kv_comm.send_recv_kv(k, v)

        if step <= kv_comm.rank or not causal:
            bwd_causal = causal and step == 0

            """
            Currently not efficient, manual backprop and this can explode the memory because of quadratic.
            """

            dsoftmax = dout * out - (dout * out).sum(dim=-1, keepdim=True) * out
            if bwd_causal:
                causal_mask = torch.tril(torch.ones((L, L), device=dsoftmax.device, dtype=dsoftmax.dtype))
                dsoftmax = dsoftmax * causal_mask
            
            block_dq_buffer = dsoftmax @ k.transpose(-2, -1)
            block_dk_buffer = q.transpose(-2, -1) @ dsoftmax
            block_dv_buffer = dsoftmax @ v

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
    ):
        if scale is None:
            scale = q.shape[-1] ** (-0.5)

        k = k.contiguous()
        v = v.contiguous()
        out, lse = _forward(group, q, k, v, scale=scale)
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.scale = scale
        ctx.causal = causal
        ctx.group = group
        return out, lse

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = _backward(
            ctx.group,
            dout,
            q,
            k,
            v,
            out,
            lse,
            scale=ctx.scale,
            causal=ctx.causal,
        )
        return dq, dk, dv, None, None, None

def ring_flex_attn(
    q,
    k,
    v,
    scale=None,
    causal=False,
    group=None,
):
    return RingFlexAttnFunc.apply(q, k, v, scale, causal, group)