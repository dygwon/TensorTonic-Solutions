import numpy as np

def matrix_inverse(A: list) -> np.ndarray | None:
    """
    Returns the inverse as a NumPy array, or None.
    """
    A = np.array(A, dtype=np.float64)

    aug = np.concatenate((A, np.eye(A.shape[0])), axis=1)

    for c in range(A.shape[0]):
        r = c + np.argmax(np.abs(aug[c:, c]))

        if np.abs(aug[r, c]) < 1e-12:
            return None

        aug[[c, r]] = aug[[r, c]]
        aug[c] = aug[c] / aug[c, c]
        mask = np.arange(A.shape[0]) != c
        aug[mask] = aug[mask] - aug[mask, c][:, None] * aug[c]

    return aug[:, A.shape[1]:]