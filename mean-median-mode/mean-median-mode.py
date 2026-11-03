import statistics

def mean_median_mode(x: list) -> dict:
    """
    Returns a dictionary with mean, median, and mode.
    """
    mean = float(statistics.mean(x))
    median = float(statistics.median(x))
    mode = float(statistics.mode(x))

    return {"mean": mean, "median": median, "mode": mode}