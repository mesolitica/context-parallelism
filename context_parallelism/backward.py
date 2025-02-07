import torch

kernel_options = {'PRESCALE_QK': False, 'ROWS_GUARANTEED_SAFE': False, 'BLOCKS_ARE_CONTIGUOUS': False, 'OUTPUT_LOGSUMEXP': True}

class FW(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, child : torch.Tensor, child_1 : torch.Tensor, child_2 : torch.Tensor, child_3 : torch.Tensor, child_4 : torch.Tensor):
        return child

class Joint(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1):
        return [arg5_1, None, None, None, None]

# ref from https://github.com/pytorch/pytorch/blob/main/torch/_higher_order_ops/flex_attention.py#L763
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
    l = query.shape[-2]
    logsumexp = logsumexp * math.log(2)
    grad_logsumexp = grad_logsumexp / math.log(2)
    scores = query.to(torch.float32) @ key.to(torch.float32).transpose(-2, -1) * scale

    if causal:
        mask = torch.triu(torch.ones(l, l), diagonal=1).bool()
        scores = scores.masked_fill(mask.to(scores.device), float('-inf'))

    masked_out_rows = logsumexp == -float('inf')
    softmax_scores = torch.exp(post_mod_scores - logsumexp.unsqueeze(-1))
    softmax_scores = torch.where(masked_out_rows.unsqueeze(-1), 0, softmax_scores)
    
    grad_value = softmax_scores.to(query.dtype).transpose(-2, -1) @ grad_out
    grad_query = None
    grad_key = None
    return grad_query, grad_key, grad_value

