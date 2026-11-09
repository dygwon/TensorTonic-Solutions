import numpy as np

def summarize(data: list, axis: int) -> np.ndarray:
    """
    Returns float64 rows of mean, standard deviation, minimum, and maximum.
    """
    data = np.array(data, dtype=np.float64)

    return np.stack([
        np.mean(data, axis=axis),
        np.std(data, axis=axis),
        np.min(data, axis=axis),
        np.max(data, axis=axis),
    ])
