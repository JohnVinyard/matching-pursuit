def safe_divide(numerator, denominator, e=1e-8):
    return numerator / (denominator + e)