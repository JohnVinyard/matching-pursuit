import zounds
from config.experiment import Experiment
from modules.pos_encode import hard_pos_encoding
from util import device
from util.readmedocs import readme


exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)


@readme
class HardPosEncodingExperiment(object):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream

        self.pos = hard_pos_encoding(exp.n_frames, device)
    
    def hard_pos(self):
        return self.pos.data.cpu().numpy().reshape((-1, exp.n_frames))
    
    def run(self):
        for i, item in enumerate(self.stream):
            item = item.view(-1, 1, exp.n_samples)