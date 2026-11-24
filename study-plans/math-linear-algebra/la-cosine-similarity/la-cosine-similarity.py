import numpy as np

def cosine_similarity(a: list, b: list) -> float:
    """
    Returns the cosine similarity as a float.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    num = np.dot(a, b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    res = num / denom
    if np.isnan(res):
        return 0.0

    return res.item()