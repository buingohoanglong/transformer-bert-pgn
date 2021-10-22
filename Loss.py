import torch
import torch.nn as nn
import sys

EPSILON = sys.float_info.epsilon

class Loss(nn.Module):
    def __init__(self, ignore_idx=None, smoothing=0.0):
        super(Loss, self).__init__()
        self.ignore_idx = ignore_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - self.smoothing

    def forward(self, pred, target):
        """
            pred: softmax dist [batch_size, seq_len, num_cls]
            target: [batch_size, seq_len]
        """
        num_cls = pred.size(-1)
        true_dist = torch.zeros_like(pred)
        num_cls_filled = num_cls - 2 if self.ignore_idx is not None else num_cls - 1
        true_dist.fill_(self.smoothing / num_cls_filled)
        true_dist.scatter_(-1, target.unsqueeze(-1), self.confidence)
        if self.ignore_idx is not None:
            true_dist[:,:,self.ignore_idx] = 0.0
            mask = (target == self.ignore_idx).unsqueeze(-1)
            true_dist.masked_fill_(mask, 0.0)
        pred = pred + torch.ones_like(pred) * EPSILON
        return torch.mean(torch.sum(-true_dist * pred.log(), dim=-1))
