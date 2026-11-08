import numpy as np

def original_and_clipped(data: list, row_idx: int, lo: float, hi: float) -> np.ndarray:
    """
    Returns a (2, n) float64 array: original row, then clipped row.
    """
    arr = np.array(data, dtype=np.float64)
    arr = arr[row_idx]
    clipped = np.clip(arr, lo, hi)

    return np.stack([arr, clipped])
