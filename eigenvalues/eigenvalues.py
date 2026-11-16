import numpy as np

def calculate_eigenvalues(matrix: list) -> np.ndarray:
    """
    Returns a sorted NumPy array of real eigenvalues.
    """
    matrix = np.array(matrix, dtype=np.float64)

    ev = np.linalg.eigvals(matrix)

    return np.sort(ev.real)