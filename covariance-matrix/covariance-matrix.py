import numpy as np

def covariance_matrix(X: list) -> np.ndarray:
    """
    Returns the covariance matrix as a NumPy array.
    """
    X = np.array(X, dtype=np.float64)
    X_c = X - X.mean(axis=0)

    return (X_c.T @ X_c) / (X.shape[0] - 1)