from torch.optim import Adam

def optimizer(model, lr=1e-4, betas=(0, 0.9), weight_decay=0):
    return Adam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)