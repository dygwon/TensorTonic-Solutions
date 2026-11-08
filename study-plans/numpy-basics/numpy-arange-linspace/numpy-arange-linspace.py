import numpy as np

def create_sequence(start: float, stop: float, param: float, kind: str) -> np.ndarray:
    """
    Returns a 1D float64 array containing the requested sequence.
    """
    if kind == "arange":
        return np.arange(start, stop, param, dtype=np.float64)
    return np.linspace(start, stop, param, dtype=np.float64)