import numpy as np

def vector_norms(v: list) -> np.ndarray:
    """
    Returns a float64 array containing the L1, L2, and infinity norms.
    """
    v = np.asarray(v, dtype=np.float64)

    l1 = np.linalg.norm(v, ord=1)
    l2 = np.linalg.norm(v, ord=None)
    inf = np.linalg.norm(v, ord=np.inf)

    return np.array([l1, l2, inf])