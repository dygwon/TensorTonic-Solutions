import numpy as np

def one_hot(y: list, num_classes=None) -> np.ndarray:
    """
    Returns a NumPy array with shape (N, K).
    """
    if num_classes is None:
        k = max(y) + 1
    else:
        k = num_classes

    y = np.asarray(y, dtype=np.int64)
    encoded = np.zeros((len(y), k), dtype=np.float64)
    encoded[np.arange(len(y)), y] = 1.0
    
    return encoded
