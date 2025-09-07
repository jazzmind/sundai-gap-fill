import torch
import torch.nn as nn
def spectral_angle_mapper(pred, target, eps=1e-6):
    p = pred.permute(0,2,3,1).reshape(-1, pred.size(1))
    t = target.permute(0,2,3,1).reshape(-1, target.size(1))
    num = (p*t).sum(dim=1)
    den = torch.norm(p,dim=1)*torch.norm(t,dim=1) + eps
    ang = torch.acos(torch.clamp(num/den, -1.0, 1.0))
    return ang.mean()
class L1SAMLoss(nn.Module):
    def __init__(self, w_l1=0.8, w_sam=0.2):
        super().__init__(); self.l1 = nn.L1Loss(); self.w_l1 = w_l1; self.w_sam = w_sam
    def forward(self, pred, target):
        return self.w_l1*self.l1(pred, target) + self.w_sam*spectral_angle_mapper(pred, target)
