import numpy as np

def row_summary(data: list, threshold: float) -> np.ndarray:
    """
    Returns a float64 array of shape (3, m, n): mask, any-row, all-row.
    """
    arr = np.array(data, dtype=np.float64)
    one = arr.copy()
    one = np.where(one > threshold, 1.0, 0.0)

    two = arr.copy()
    mask = np.expand_dims(np.any(two > threshold, axis=1), -1)
    two = np.where(mask, two, np.zeros(two.shape[1]))

    three = arr.copy()
    mask = np.expand_dims(np.all(three > threshold, axis=1), -1)
    three = np.where(mask, three, np.zeros(three.shape[1]))

    return np.stack([one, two, three])