from cmath import isnan
import numpy as np


def waveguide_synth(
    impulse: np.ndarray, 
    delay: np.ndarray, 
    damping: np.ndarray, 
    filter_size: int) -> np.ndarray:
    
    n_samples = impulse.shape[0]

    output = impulse.copy()
    buf = np.zeros_like(impulse)

    for i in range(n_samples):
        delay_val = 0

        delay_amt = delay[i]
        
        if i > delay_amt:
            damping_amt = damping[i]
            delay_val += output[i - delay_amt] * damping_amt

        
        buf[i] = delay_val

        filt_size = filter_size[i]        
        filt_slice = buf[i - filt_size: i]   
        if filt_slice.shape[0]:
            new_val = np.mean(filt_slice)
        else:
            new_val = delay_val

        output[i] += new_val
    
    return output

