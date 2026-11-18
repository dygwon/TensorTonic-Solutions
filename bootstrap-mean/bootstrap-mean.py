import numpy as np

def bootstrap_mean(x: list, n_bootstrap: int = 1000, ci: float = 0.95, seed: int = 0) -> dict:
    """
    Returns a dictionary with bootstrap_mean, lower, and upper.
    """
    x = np.array(x, dtype=np.float64)
    rng = np.random.default_rng(seed)
    alpha = (1 - ci) / 2

    indices = rng.integers(0, x.size, size=(n_bootstrap, x.size))
    means = x[indices].mean(axis=1)
    bootstrap_mean = means.mean()

    return {"bootstrap_mean": bootstrap_mean, "lower": np.quantile(means, alpha), "upper": np.quantile(means, 1-alpha)}