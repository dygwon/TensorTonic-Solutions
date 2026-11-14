import math

def binomial_pmf_cdf(n: int, p: float, k: int) -> dict:
    """
    Returns a dictionary with pmf and cdf.
    """
    
    def formula(n: int, k: int) -> float:
        return float(math.comb(n, k) * p**k * (1 - p)**(n - k))

    pmf = formula(n, k)    
    
    cdf = 0.0
    for i in range(k + 1):
        cdf += formula(n, i)
        
    return {"pmf": pmf, "cdf": cdf}