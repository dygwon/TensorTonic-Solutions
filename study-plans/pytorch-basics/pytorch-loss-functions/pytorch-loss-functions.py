import torch

def compute_loss(pred: torch.Tensor, target: torch.Tensor, method: str, delta: float = 1.0) -> float:
    """
    Returns the mean loss as a Python float.
    """
    match method:
        case "mse":
            return torch.mean((pred - target)**2).item()
        case "cross_entropy":
            maximums = pred.max(dim=-1, keepdim=True).values
            log_partition = (pred - maximums).exp().sum(dim=-1).log()
            res = log_partition - (pred[torch.arange(pred.shape[0]), target] - maximums.squeeze(-1))
            return torch.mean(res).item()
        case "huber":
            a = pred - target
            res = torch.where(torch.abs(a) <= delta, a**2 / 2, delta * torch.abs(a) - delta**2 / 2)
            return torch.mean(res).item()
