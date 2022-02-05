
from config.dotenv import Config
from data.datastore import batch_stream
import torch
from modules.psychoacoustic import PsychoacousticFeature
from util import device
from torch import nn
from torch.optim import Adam
import zounds
from torch.nn import functional as F

n_samples = 2 ** 14

class ParamWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(
            torch.FloatTensor(1, 1, n_samples).uniform_(-0.01, 0.01))
    
    def forward(self, x):
        return self.p


def real():
        return zounds.AudioSamples(samples.squeeze(), sr).pad_with_silence()
    
def fake():
    return zounds.AudioSamples(model.p.data.cpu().numpy().squeeze(), sr).pad_with_silence()

if __name__ == '__main__':
    app = zounds.ZoundsApp(locals=locals(), globals=globals())
    app.start_in_thread(9999)

    sr = zounds.SR22050()

    stream = batch_stream(Config.audio_path(), '*.wav', 1, n_samples)
    samples = next(stream)

    s = torch.from_numpy(samples).float().to(device)

    model = ParamWrapper().to(device)
    optim = Adam(model.parameters(), lr=1e-4, betas=(0, 0.9))

    feature = PsychoacousticFeature().to(device)

    while True:
        optim.zero_grad()

        real_feat = feature.compute_feature_dict(
            s, constant_window_size=128)
        recon_feat = feature.compute_feature_dict(
            model(None), constant_window_size=128)

        loss = 0
        for k, v in real_feat.items():
            loss = loss + F.mse_loss(recon_feat[k], v)
        
        loss.backward()
        optim.step()

        print(loss.item())




