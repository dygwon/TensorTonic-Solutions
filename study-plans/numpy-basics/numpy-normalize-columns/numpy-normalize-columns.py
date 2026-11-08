import numpy as np

def normalize(data: list) -> np.ndarray:
    """
    Returns a float64 matrix standardized independently by column.
    """
    X = np.array(data, dtype=np.float64)
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)

    return (X - mu) / sigma
