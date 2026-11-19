import torch

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask = None,
) -> torch.Tensor:
    """
    Returns a float32 attention tensor with shape (batch, query length, value width).
    """
    inner = (query @ key.transpose(1, 2))
    dk = torch.full_like(inner, query.shape[2])
    inner = inner / torch.sqrt(dk)
    if mask is not None:
        inner = inner.masked_fill(mask, -float("inf"))
    return torch.softmax(inner, dim=-1) @ value
