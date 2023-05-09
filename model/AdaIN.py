import torch
from torch import nn

class AdaIN(nn):
    def __init__(self, lantent, channel):
        self.linear_mean = nn.Linear(lantent, channel)
        self.linear_std = nn.Linear(lantent, channel)

    def _calc_mean_std(self, feat: torch.Tensor, eps = 1e-5):
        """
        eps is a small value added to the variance to avoid divide-by-zero.
        """
        size = feat.size()
        N, C = size[: 2]
        feat_var = feat.view(N, C, -1).var(dim = 2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim = 2).view(N, C, 1, 1)
        return feat_mean, feat_std