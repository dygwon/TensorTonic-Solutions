import numpy as np

def matrix_multiply(A: list, B: list) -> np.ndarray:
    """
    Returns the matrix product as a float64 array.
    """
    return np.asarray(A, dtype=np.float64) @ np.asarray(B, dtype=np.float64)