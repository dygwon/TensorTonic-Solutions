import numpy as np

def activation_derivative(name: str, x: list) -> list:
    """
    Returns one analytical derivative for each input value.
    """
    x = np.asarray(x, dtype=np.float64)
    match name:
        case "sigmoid":
            sig = 1 / (1 + np.exp(-x))
            return list(sig * (1 - sig))
        case "tanh":
            return list(1 - np.tanh(x)**2)
        case "relu":
            return list(np.where(x > 0, 1.0, 0.0))
        case "swish":
            sig = 1 / (1 + np.exp(-x))
            return list(sig + x * sig * (1 - sig))