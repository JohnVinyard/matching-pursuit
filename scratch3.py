import torch
import torch.nn.functional as F
from modules.stft import stft
from matplotlib import pyplot as plt


from modules.transfer import freq_domain_transfer_function_to_resonance



if __name__ == '__main__':
    window_size = 1024
    step_size = window_size // 2
    n_coeffs = window_size // 2 + 1
    
    coeffs = torch.zeros(n_coeffs).uniform_(0, 1)
    
    coeffs = coeffs[None, :]
    
    audio = freq_domain_transfer_function_to_resonance(window_size, coeffs, 128)
    spec = stft(audio, 512, 256, pad=True)
    spec = spec.data.cpu().numpy().squeeze()
    
    plt.matshow(spec)
    plt.show()


    