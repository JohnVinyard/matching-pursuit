import torch
import zounds
import numpy as np
from modules.normal_pdf import pdf
from modules.recurrent import RecurrentSynth


if __name__ == '__main__':

    app = zounds.ZoundsApp(globals=globals(), locals=locals())
    app.start_in_thread(9999)

    rec = RecurrentSynth(3, 128)
    start = torch.zeros(1, 128).normal_(0, 1)

    hidden, gate = rec.forward(start, max_iter=20)
    hidden = hidden.data.cpu().numpy().squeeze().T
    gate = gate.data.cpu().numpy().squeeze()
    print(hidden.shape, gate.shape)

    input('Waiting...')
