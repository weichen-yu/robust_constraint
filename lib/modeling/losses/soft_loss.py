import torch
import torch.nn.functional as F
from torch import nn
from .base import BaseLoss, gather_and_scale_wrapper

class SoftLoss(BaseLoss):
    def __init__(self, loss_term_weight=1.0):
        super(SoftLoss, self).__init__(loss_term_weight)
        self.loss_info = {}

    @gather_and_scale_wrapper
    def forward(self, x, target, mask, temperature):
        target = F.softmax(target/temperature, dim=-1)
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1) * mask
        self.loss_info['scalar/softmax/soft_loss'] = loss.mean(dim=0)
        return loss.mean(dim=0), self.loss_info
