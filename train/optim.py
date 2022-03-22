from torch.optim import Adam

def optimizer(model, lr=1e-4, betas=(0, 0.9)):
    return Adam(model.parameters(), lr=lr, betas=betas)