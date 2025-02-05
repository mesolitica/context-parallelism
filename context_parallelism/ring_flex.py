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
    causal=True,
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
):
    kv_comm = RingComm(process_group)
    d_kv_comm = RingComm(process_group)
    dq, dk, dv = None, None, None
    next_dk, next_dv = None, None

    block_dq_buffer = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    block_dk_buffer = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    block_dv_buffer = torch.empty(v.shape, dtype=v.dtype, device=v.device)

    next_dk, next_dv = None, None
    next_k, next_v = None, None
    

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
        out, lse = _forward(
            group,
            q,
            k,
            v,
            scale=scale,
        )
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.scale = scale
        ctx.causal = causal
        ctx.group = group
        return out, lse

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, lse = ctx.saved_tensors
        # dq, dk, dv = ring_flash_attn_backward(
        #     ctx.group,
        #     dout,
        #     q,
        #     k,
        #     v,
        #     out,
        #     softmax_lse,
        #     softmax_scale=ctx.softmax_scale,
        #     dropout_p=ctx.dropout_p,
        #     causal=ctx.causal,
        #     window_size=ctx.window_size,
        #     alibi_slopes=ctx.alibi_slopes,
        #     deterministic=ctx.deterministic,
        # )
        return dq, dk, dv, None, None, None

def ring_flex_attn(
    q,
    k,
    v,
    scale=None,
    causal=False,
    group=None,
):
    return RingFlexAttnFunc.apply(
        q,
        k,
        v,
        scale,
        causal,
        group,
    )