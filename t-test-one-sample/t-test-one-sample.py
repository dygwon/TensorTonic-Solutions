import numpy as np

def t_test_one_sample(x: list, mu0: float) -> float:
    """
    Returns the t-statistic as a float.
    """
    x = np.array(x, dtype=np.float64)

    s = np.std(x, ddof=1)
    res = (np.mean(x) - mu0) / (s / np.sqrt(x.shape[0]))
    return res.item()