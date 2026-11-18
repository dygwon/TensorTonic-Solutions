import numpy as np

def zscore_standardize(X: list, axis: int = 0, eps: float = 1e-12) -> np.ndarray:
    """
    Returns population Z-scores as a NumPy array matching the shape of X.
    """
    X = np.array(X, dtype=np.float64)

    means = X.mean(axis=axis, keepdims=True)
    stds = X.std(axis=axis, keepdims=True)

    return (X - means) / np.where(stds > eps, stds, 1.0)