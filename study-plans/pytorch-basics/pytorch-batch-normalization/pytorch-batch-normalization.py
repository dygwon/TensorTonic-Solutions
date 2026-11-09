import torch

def batch_norm(X: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Returns a float32 tensor with the same shape as X.
    """
    means = X.mean(dim=0)
    vars = torch.mean((X - means)**2, dim=0)

    return (X - means) / torch.sqrt(vars + eps) * gamma + beta
