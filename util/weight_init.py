import torch

def make_initializer(init_value):
    def init_weights(p):

        with torch.no_grad():
            try:
                p.weight.uniform_(-init_value, init_value)
            except AttributeError:
                pass

            try:
                p.bias.fill_(0)
            except AttributeError:
                pass
    return init_weights
