from torch import nn
from torch.nn.utils.weight_norm import weight_norm
import torch.nn.functional as F
import torch

# class MixerBlock(nn.Module):
#     def __init__(self, channels, size):
#         super().__init__()
#         self.channels = channels
#         self.size = size

#         self.l1 = nn.Linear(channels, size)
#         self.l1_gate = nn.Linear(channels, size)


#         self.l2 = nn.Linear(size, channels)
#         self.l2_gate = nn.Linear(channels, size)
    
#     def forward(self, x):
#         residual = x
#         x = x.view(-1, self.size, self.channels)
#         x = self.l1(x) # (-1, size, size)
#         x = x.permute(0, 2, 1)
#         x = self.l2(x) # (-1, size, channels)
#         x = x + residual
#         return x


# class Mixer(nn.Module):

#     def __init__(self, channels, size, layers, return_features=False):
#         super().__init__()
#         self.channels = channels
#         self.size = size
#         self.layers = layers
#         self.return_features = return_features

#         self.net = nn.Sequential(*[
#             MixerBlock(channels, size)
#             for _ in range(layers)
#         ])
    
#     def forward(self, x):
#         features = []
#         for layer in self.net:
#             x = layer(x)
#             features.append(x)
#             x = F.leaky_relu(x, 0.2)
        
#         if self.return_features:
#             return x, features
#         else:
#             return x
        
        


class MixerBlock(nn.Module):
    def __init__(self, channels, sequence_length):
        super().__init__()
        self.channels = channels
        self.sequence_length = sequence_length

        self.pos = nn.Parameter(torch.zeros(1, sequence_length, channels).uniform_(-0.01, 0.01))

        self.proj1 = weight_norm(nn.Linear(channels, channels))
        self.proj2 = weight_norm(nn.Linear(sequence_length, channels))
        self.proj3 = weight_norm(nn.Linear(channels, sequence_length))
        self.nl = lambda x: F.elu(x)

    
    def forward(self, x):


        assert x.shape[1:] == (self.sequence_length, self.channels)

        x = F.dropout(x, 0.1)

        skip = x

        tr = x.permute(0, 2, 1) # (Batch, channels, seq_len)
        tr = self.proj2(tr) # (Batch, channels, channels)
        tr = self.proj3(tr) # (Batch, channels, seq_len)
        tr = tr.permute(0, 2, 1) # (Batch, seq_len, channels)

        x = self.proj1(x + self.pos) # (Batch, seq_len, channels)

        x = x + tr + skip
        x = self.nl(x)


        return x


class MixerAttention(nn.Module):
    def __init__(self, channels, sequence_length, n_modules):
        super().__init__()
        self.blocks = nn.ModuleList([MixerBlock(channels, sequence_length) for _ in range(n_modules)])

        self.channels = channels
        self.sequence_length = sequence_length
        self.n_modules = n_modules

        self.down1 = weight_norm(nn.Linear(channels, 1))
        self.down2 = weight_norm(nn.Linear(sequence_length, n_modules))
    
    def forward(self, x):

        # (batch, seq_len, channels) => (Batch, n_modules)
        attn = self.down1(x).view(-1, self.sequence_length)
        attn = self.down2(attn).view(-1, self.n_modules, 1, 1)
        attn = torch.softmax(attn, dim=1)

        outputs = [block(x)[:, None, :, :] for block in self.blocks]
        outputs = torch.cat(outputs, dim=1) # (Batch, modules, seq_len, channels)

        final = outputs * attn
        final = torch.sum(final, dim=1)
        return final

        

class MixerStack(nn.Module):
    def __init__(self, in_channels, channels, sequence_length, layers, attn_blocks, channels_last=True):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.sequence_length = sequence_length
        self.layers = layers
        self.channels_last = channels_last

        self.net = nn.Sequential(
            nn.Linear(in_channels, channels),
            *[MixerAttention(channels, sequence_length, attn_blocks) for _ in range(layers)],
            nn.Linear(channels, channels)
        )
    
    def forward(self, x):
        if not self.channels_last:
            x = x.permute(0, 2, 1)
        
        x = self.net(x)

        if not self.channels_last:
            x = x.permute(0, 2, 1)
        return x