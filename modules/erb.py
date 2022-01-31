def erb(f, sr):
    f = f * sr.nyquist
    return (0.108 * f) + 24.7

def scaled_erb(f, sr):
    return erb(f, sr) / sr.nyquist