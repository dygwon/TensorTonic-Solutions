import torch

def softmax(logits: torch.Tensor) -> torch.Tensor:
    """
    Returns a float32 probability tensor with the same shape as logits.
    """
    logits = logits - logits.max(dim=1, keepdim=True).values
    return logits.exp() / logits.exp().sum(dim=1, keepdim=True)
