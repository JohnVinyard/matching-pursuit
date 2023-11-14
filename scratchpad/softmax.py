import torch
from torch.nn import functional as F


if __name__ == '__main__':
    t = torch.zeros(1000, 128).normal_(0, 1)

    a = torch.softmax(t, dim=-1)
    b = F.gumbel_softmax(torch.exp(t), dim=-1, tau=1e-6, hard=False)

    a = torch.argmax(a, dim=-1)
    b = torch.argmax(b, dim=-1)
    # c = torch.argmax(t, dim=-1)

    # print(a)
    # print(b)
    # print(c)

    print((a == b).sum().item() / t.shape[0])