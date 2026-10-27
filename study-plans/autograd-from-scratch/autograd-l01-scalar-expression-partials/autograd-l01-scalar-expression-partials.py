def scalar_expression_partials(a: float, b: float, c: float, h: float) -> tuple[float, float, float, float]:
    """
    Returns the expression value and numerical partials for a, b, and c.
    """
    def d(a: float, b: float, c: float) -> float:
        return float(a * b + c)

    dd = d(a, b, c)
    da = d(a + h, b, c)
    db = d(a, b + h, c)
    dc = d(a, b, c + h)

    return dd, (da - dd) / h, (db - dd) / h, (dc - dd) / h
    