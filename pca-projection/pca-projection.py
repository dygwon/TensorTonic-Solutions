import numpy as np

def pca_projection(X: list, k: int) -> list:
    """
    Returns the centered data projected onto the top components.
    """
    X = np.array(X, dtype=np.float64)
    centered = X - X.mean(axis=0)
    cov = (centered.T @ centered) / (X.shape[0] - 1)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    indices = np.argsort(eigenvalues)[::-1][:k]
    eigenvectors = eigenvectors[:, indices]

    return centered @ eigenvectors
