import torch
from torch.nn import functional as F


if __name__ == '__main__':
    t = torch.zeros(100, 128).normal_(0, 1)

    a = torch.softmax(t, dim=-1)
    b = F.gumbel_softmax(torch.exp(t), dim=-1, tau=1, hard=True)

    a = torch.argmax(a, dim=-1)
    b = torch.argmax(b, dim=-1)
    c = torch.argmax(t, dim=-1)

    print(a)
    print(b)
    print(c)