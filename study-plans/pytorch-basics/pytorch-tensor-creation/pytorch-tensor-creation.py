import torch

def create_tensor(method: str, shape: list, value: float = 0.0) -> torch.Tensor:
    """
    Returns a float32 tensor with the requested shape.
    """
    res = None
    match method:
        case "zeros":
            res = torch.zeros(*shape, dtype=torch.float32)
        case "ones":
            res = torch.ones(*shape, dtype=torch.float32)
        case "full":
            res = torch.full(shape, value, dtype=torch.float32)

    return res
