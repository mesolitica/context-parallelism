import torch
import math
from .utils import causal_mask
from torch.nn.attention.flex_attention import (
    create_block_mask,
    _identity, 
    _create_empty_block_mask, 
    _apply_kernel_options,
)
from torch._higher_order_ops.flex_attention import (
    sdpa_dense_backward, 
    create_fw_bw_graph,
)

def attention_backward(
    query,
    key,
    value,
    out,
    logsumexp,
    grad_out,
    grad_logsumexp,
    scale,
    causal=False,
):
    kernel_options = _apply_kernel_options(
        query,
        key,
        value,
        True,
        None,
    )
    if causal:
        block_mask = create_block_mask(
            causal_mask, 
            None, 
            None, 
            query.shape[-2], 
            query.shape[-2], 
        )
    else:
        block_mask = _create_empty_block_mask(q, k)

    block_mask = block_mask.as_tuple()
    example_vals = (
        query.new_zeros((), requires_grad=True),
        query.new_zeros((), dtype=torch.int),
        query.new_zeros((), dtype=torch.int),
        query.new_zeros((), dtype=torch.int),
        query.new_zeros((), dtype=torch.int),
    )
    fw_graph, bw_graph = create_fw_bw_graph(
        _identity, example_vals, (),
    )
    """
    https://github.com/pytorch/pytorch/blob/main/torch/_higher_order_ops/flex_attention.py#L763
    sdpa_dense_backward(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        out: torch.Tensor,
        logsumexp: torch.Tensor,
        grad_out: torch.Tensor,
        grad_logsumexp: torch.Tensor,
        fw_graph: Callable,
        joint_graph: Callable,
        block_mask: Tuple,
        scale: float,
        kernel_options: Dict[str, Any],
        score_mod_other_buffers: Tuple,
        mask_mod_other_buffers: Tuple,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[Optional[torch.Tensor], ...]]
    """
    o = sdpa_dense_backward(
        query,
        key,
        value,
        out,
        logsumexp,
        grad_out,
        grad_logsumexp,
        fw_graph,
        bw_graph,
        block_mask, 
        scale, 
        kernel_options,
        score_mod_other_buffers = (),
        mask_mod_other_buffers = (),
    )
    return o[:-1]




