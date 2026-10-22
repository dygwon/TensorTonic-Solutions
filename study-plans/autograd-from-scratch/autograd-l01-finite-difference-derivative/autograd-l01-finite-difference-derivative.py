import numpy as np

def finite_difference_derivative(coefficients: list, x: float, h: float) -> tuple[float, float, float]:
    """
    Returns the value at x, the value at x plus h, and the estimated slope.
    """
    fx = np.polynomial.polynomial.polyval(x, coefficients).item()
    fxh = np.polynomial.polynomial.polyval(x + h, coefficients).item()
    slope = (fxh - fx) / h

    return fx, fxh, slope
