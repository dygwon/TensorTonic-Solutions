import numpy as np

def cosine_similarity(a: list, b: list) -> float:
    """
    Returns the cosine similarity as a Python float.
    """
    res = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    if np.isnan(res):
        res = 0.0

    return res