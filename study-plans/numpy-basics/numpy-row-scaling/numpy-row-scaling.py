import numpy as np

def scale_rows(data: list, weights: list) -> np.ndarray:
    """
    Returns a float64 matrix with each row multiplied by its weight.
    """
    X = np.array(data, dtype=np.float64)
    w = np.array(weights, dtype=np.float64)

    return X * w[:, np.newaxis]
