import numpy as np

def matrix_vector_multiply(A: list, x: list) -> np.ndarray:
    """
    Returns the matrix-vector product as a float64 array.
    """
    return np.asarray(A, dtype=np.float64) @ np.asarray(x, dtype=np.float64)