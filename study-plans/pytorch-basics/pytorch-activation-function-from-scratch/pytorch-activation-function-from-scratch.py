import torch

def activate(x: torch.Tensor, method: str = "relu") -> torch.Tensor:
    """
    Returns a float32 tensor with the same shape as x.
    """
    match method:
        case "relu":
            return torch.clamp(x, min=0)
        case "sigmoid":
            return 1 / (1 + torch.exp(-x))
        case "tanh":
            ex = torch.exp(-2 * torch.abs(x))
            s = torch.where(x < 0, -1, 1)
            return s * (1 - ex) / (1 + ex)
        case "leaky_relu":
            return torch.where(x > 0, x, 0.01 * x)
