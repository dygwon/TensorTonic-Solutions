import numpy as np

def angle_features(angles: list) -> np.ndarray:
    """
    Returns a (3, n) float64 array with sine, cosine, and tangent rows.
    """
    angles = np.array(angles, dtype=np.float64)

    return np.stack([np.sin(angles), np.cos(angles), np.tan(angles)])
