import numpy as np

def filter_and_extract(data: list, row_start: int, row_stop: int, threshold: float) -> np.ndarray:
    """
    Returns matching values in row-major order as a 1D float64 array.
    """
    arr = np.array(data, dtype=np.float64)[row_start : row_stop, :]
    return arr[arr > threshold].flatten()
