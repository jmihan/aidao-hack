import torch

def mae(output, target):
    return torch.mean(torch.abs(output - target)).item()

def mse(output, target):
    return torch.mean((output - target)**2).item()