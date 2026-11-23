import numpy as np

def one_hot(y: list, num_classes=None) -> np.ndarray:
    """
    Returns a NumPy array with shape (N, K).
    """
    if num_classes is None:
        k = max(y) + 1
    else:
        k = num_classes

    res = []
    for yi in y:
        row = np.zeros(k)
        row[yi] = 1.0

        res.append(row)

    return np.array(res, dtype=np.float64)