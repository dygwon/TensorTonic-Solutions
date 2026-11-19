import torch

def multi_head_attention(
    hidden_states: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    num_heads: int,
    causal: bool = False,
) -> torch.Tensor:
    """
    Returns a float32 tensor with the same shape as hidden_states.
    """
    q = hidden_states @ w_q
    k = hidden_states @ w_k
    v = hidden_states @ w_v

    d_h = int(hidden_states.shape[-1] / num_heads)
    q = q.reshape((q.shape[0], q.shape[1], num_heads, d_h))
    k = k.reshape((k.shape[0], k.shape[1], num_heads, d_h))
    v = v.reshape((v.shape[0], v.shape[1], num_heads, d_h))
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    score = q @ k.transpose(-1, -2) / d_h**(0.5)
    mask = torch.triu(score, diagonal=1) if causal else torch.zeros_like(score)
    score = torch.where(mask == 0.0, score, -float("inf"))
    h = torch.softmax(score, dim=-1) @ v
    h = h.transpose(1, 2)
    h = h.reshape((h.shape[0], h.shape[1], hidden_states.shape[-1]))

    return h @ w_o