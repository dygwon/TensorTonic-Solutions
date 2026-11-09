import numpy as np

def sort_with_indices(data: list, axis: int) -> np.ndarray:
    """
    Returns a (2, m, n) float64 array of sorted values and source indices.
    """
    data = np.array(data, dtype=np.float64)

    return np.stack([
        np.sort(data, axis=axis),
        np.argsort(data, axis=axis),
    ])
