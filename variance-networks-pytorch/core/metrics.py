import numpy as np
import torch.nn.functional as F
from torch import nn
import torch
from torch.autograd import Variable


class SGVLB(nn.Module):
    def __init__(self, net, train_size):
        super(SGVLB, self).__init__()
        self.train_size = train_size
        self.net = net

    def forward(self, input, target, kl_weight=1.0):
        assert not target.requires_grad
        kl = 0.0
        for module in self.net.modules():
            if hasattr(module, 'kl_reg'):
                kl = kl + module.kl_reg()
        return F.cross_entropy(input, target, reduction='elementwise_mean') * self.train_size + kl_weight * kl

    def get_kl(self):
        kl = 0.0
        for module in self.net.modules():
            if hasattr(module, 'kl_reg'):
                kl = kl + module.kl_reg()
        return kl


def lr_linear(epoch_num, decay_start, total_epochs, start_value):
    if epoch_num < decay_start:
        return start_value
    return start_value*float(total_epochs-epoch_num)/float(total_epochs-decay_start)


def correct(output, target, topk=(1,)):
    """Computes the correct@k for the specified values of k"""
    maxk = max(topk)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t().type_as(target)
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0).item()
        res.append(correct_k)
    return res

def logit2acc(outputs, targets):
    corr = correct(outputs, targets)
    return corr[0] / targets.shape[0]


def kl_ard(log_alpha):
    return 0.5 * torch.sum(torch.log1p(torch.exp(-log_alpha)))


def kl_loguni(log_alpha):
    k1, k2, k3 = 0.63576, 1.8732, 1.48695
    C = -k1
    mdkl = k1 * torch.sigmoid(k2 + k3 * log_alpha) - 0.5 * torch.log1p(torch.exp(-log_alpha)) + C
    kl = -torch.sum(mdkl)
    return kl
