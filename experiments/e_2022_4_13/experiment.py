from torch import nn
from torch.nn import functional as F
import torch
from modules import stft
from train.optim import optimizer
from util import device, playable
from util.readmedocs import readme
import zounds


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_filters = 2048
        self.filter_length = 2048
        self.k = 2048
        self.atoms = nn.Parameter(torch.zeros(self.n_filters, 1, self.filter_length).normal_(0, 0.01))
    
    def forward(self, x):
        norms = torch.norm(self.atoms, dim=-1, keepdim=True)
        atoms = self.atoms / (norms + 1e-12)
        feature_map = x = F.conv1d(x, atoms, bias=None, stride=1, padding=self.filter_length // 2)

        shape = feature_map.shape

        feature_map = feature_map.view(-1)
        values, indices = torch.topk(torch.abs(feature_map), k=self.k)
        new_feature_map = torch.zeros_like(feature_map)
        new_feature_map[indices] = values
        feature_map = new_feature_map.reshape(shape)

        x = F.conv_transpose1d(feature_map, atoms, bias=None, stride=1, padding=self.filter_length // 2)
        return feature_map, x


model = Model().to(device)
optim = optimizer(model)

def train_model(batch):
    optim.zero_grad()
    feature_map, recon = model(batch)

    real_spec = torch.log(1e-12 + stft(batch))
    fake_spec = torch.log(1e-12 + stft(recon))
    

    recon_loss = F.mse_loss(fake_spec, real_spec)


    loss = recon_loss
    loss.backward()
    optim.step()
    print(loss.item())
    return feature_map, recon

@readme
class MatchingPursuit(object):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream

        self.feature_map = None
        self.recon = None
    
    def listen(self):
        return playable(self.recon, zounds.SR22050())
    
    def look(self):
        return self.feature_map[0].data.cpu().numpy().squeeze()
    
    def run(self):
        for item in self.stream:
            item = item.view(-1, 1, 2**14)
            f, r = train_model(item)
            self.feature_map = f
            self.recon = r