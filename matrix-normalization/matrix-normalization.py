import numpy as np

def matrix_normalization(matrix: list, axis=None, norm_type: str = "l2") -> np.ndarray:
    """
    Returns a NumPy array with the same shape as matrix.
    """
    matrix = np.array(matrix, dtype=np.float64)
    match norm_type:
        case "l1":
            norm = np.sum(np.abs(matrix), axis=axis, keepdims=True)
        case "l2":
            norm = np.sum(matrix**2, axis=axis, keepdims=True)
            norm = np.sqrt(norm)
        case "max":
            norm = np.max(np.abs(matrix), axis=axis, keepdims=True)

    norm = np.where(norm == 0.0, 1.0, norm)
    return matrix / norm