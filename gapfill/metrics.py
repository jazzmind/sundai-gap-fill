import torch
from math import log10
def rmse(pred, tgt): return torch.sqrt(torch.mean((pred - tgt)**2)).item()
def mae(pred, tgt):  return torch.mean(torch.abs(pred - tgt)).item()
def psnr(pred, tgt, max_val=1.2):
    mse = torch.mean((pred - tgt)**2).item()
    return 10 * log10((max_val**2)/(mse+1e-8))
def ssim_simple(pred, tgt, C1=0.01**2, C2=0.03**2):
    mu_x = pred.mean(); mu_y = tgt.mean()
    sigma_x = pred.var(); sigma_y = tgt.var()
    sigma_xy = ((pred - mu_x)*(tgt - mu_y)).mean()
    num = (2*mu_x*mu_y + C1) * (2*sigma_xy + C2)
    den = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)
    return (num/den).item()
