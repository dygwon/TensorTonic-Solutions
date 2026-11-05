import torch

def tensor_op(x: torch.Tensor, y: torch.Tensor, op: str) -> torch.Tensor:
    """
    Returns the operation result as a float32 tensor.
    """
    res = None
    
    match op:
        case "add":
            res = torch.add(x, y)
        case "multiply":
            res = torch.mul(x, y)
        case "power":
            res = torch.pow(x, y)
        case "max":
            res = torch.maximum(x, y)
        case "matmul":
            res = torch.matmul(x, y)

    return res