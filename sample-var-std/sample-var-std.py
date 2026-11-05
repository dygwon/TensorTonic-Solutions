import numpy as np

def sample_var_std(x: list) -> dict:
    """
    Returns a dictionary with variance and standard_deviation.
    """
    # Write code here
    var = 0.0
    mu = np.mean(x, dtype=np.float32).item()
    for el in x:
        var += (el - mu)**2
    var = var / (len(x) - 1)
    sd = np.sqrt(var).item()

    return {"variance": var, "standard_deviation": sd}