import numpy as np

def bernoulli_pmf_and_moments(x: list, p: float) -> dict:
    """
    Returns a dictionary with pmf, mean, and variance.
    """
    x = np.array(x, dtype=np.float64)
    pmf = np.where(x == 1, p, 1.0 - p)
    mu = float(p)
    var = p * (1.0 - p)

    return {"pmf": pmf, "mean": mu, "variance": var}