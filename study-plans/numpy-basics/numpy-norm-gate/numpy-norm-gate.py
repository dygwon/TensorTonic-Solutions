import numpy as np

def norm_gate(X: list, W: list, threshold: float) -> np.ndarray:
    """
    Returns an (n, k) float64 matrix of norm-gated transformed rows.
    """
    X = np.array(X, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    Z = X @ W

    return np.where(np.linalg.norm(Z, axis=1, keepdims=True) >= threshold, Z, 0.0)
