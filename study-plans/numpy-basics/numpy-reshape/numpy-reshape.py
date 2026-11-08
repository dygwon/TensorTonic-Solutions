import numpy as np

def reshape_array(data: list, operation: str) -> np.ndarray:
    """
    Returns a float64 array with the shape selected by operation.
    """
    match operation:
        case "flatten":
            return np.array(data, dtype=np.float64).flatten()
        case "transpose":
            return np.array(data, dtype=np.float64).T
        case _:
            return np.expand_dims(np.array(data, dtype=np.float64), 0)
