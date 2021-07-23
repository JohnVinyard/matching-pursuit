from torch import nn
import torch
from torch.nn import functional as F


class GlobalContext(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.contr = nn.Linear(channels, 1)
        
    
    def forward(self, x):
        n_elements = x.shape[0]
        x = x.view(-1, self.channels)
        x = x[None, :] + x[:, None]
        x = x.view(n_elements, n_elements, self.channels)
        z = self.contr(x)
        x = x * z
        x = torch.sum(x, dim=1)
        return x

class Cluster(nn.Module):
    def __init__(
        self, 
        channels, 
        n_clusters, 
        aggregate=lambda x: torch.mean(x, dim=0, keepdim=True)):

        super().__init__()
        self.channels = channels
        self.n_clusters = n_clusters

        self.assign = nn.Linear(channels, n_clusters)
        self.aggregate = aggregate
        
    def forward(self, x):
        x = self.assign
        x = F.softmax(x)
        indices = torch.max(x, dim=1)

        output = torch.zeros(self.n_clusters, self.channels).to(x.device)
        for i in range(self.n_clusters):
            indx = indices == i
            output[i] = self.aggregate(x[indx])
        
        return output


class Reducer(nn.Module):
    def __init__(self, channels, factor):
        super().__init__()
        self.channels = channels
        self.factor = factor
        # TODO: How to do learned clustering with fixed cluster sizes?

    
    def forward(self, x):
        pass


if __name__ == '__main__':
    t = torch.FloatTensor(13, 5).normal_(0, 1)

    gc = GlobalContext(5)

    x = gc.forward(t)

    print(x.shape)

