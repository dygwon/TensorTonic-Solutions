import numpy as np

def pearson_correlation(X: list) -> np.ndarray:
    """
    Returns the correlation matrix as a NumPy array.
    """
    X = np.array(X, dtype=np.float64)

    centered = X - np.mean(X, axis=0)
    cov = centered.T @ centered / (X.shape[0] - 1)

    stdev = np.sqrt(np.diag(cov))
    return cov / np.outer(stdev, stdev.T)