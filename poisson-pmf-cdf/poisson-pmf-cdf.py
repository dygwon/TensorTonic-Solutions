import math

def poisson_pmf_cdf(lam: float, k: int) -> dict:
    """
    Returns a dictionary with pmf and cdf.
    """
    def prob_events(i: int) -> float:
        return float(math.exp(-lam) * lam**i / math.factorial(i))

    pmf = prob_events(k)
    
    cdf = 0.0
    for i in range(k+1):
        cdf += prob_events(i)

    return {"pmf": pmf, "cdf": cdf}