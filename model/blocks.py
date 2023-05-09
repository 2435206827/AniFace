import torch
import torch.nn.functional as F
from torch import nn

class mapping(nn):
    def __init__(self, dim = 512):
        super(mapping, self).__init__()
        self.fc1 = nn.Linear(dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 256)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.fc2(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.fc3(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.fc4(x)
        return x

class AdaIN(nn):
    """
    @ref: https://blog.csdn.net/weixin_35576881/article/details/91563347
    """

    def __init__(self, lantent, channel):
        self.linear_mean = nn.Linear(lantent, channel)
        self.linear_std = nn.Linear(lantent, channel)

    def _calc_mean_std(self, feat: torch.Tensor, eps = 1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        N, C = size[: 2]
        feat_var = feat.view(N, C, -1).var(dim = 2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim = 2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def forward(self, content_feat, style_feat):
        """
        dim: (N, C, W, H)
        """
        style_mean, style_std = self.linear_mean(style_feat), self.linear_std(style_feat)
        content_mean, content_std = self._calc_mean_std(content_feat)

        normalized_feat = (content_feat - content_mean) / content_std
        return normalized_feat * style_std + style_mean

class AFFA_module(nn):
    def __init__(self, c, s):
        """
        param:
        c is the channel of input
        s is the size of input (equals to W or H)
        """
        super(AFFA_module, self).__init__()
        self.size = s
        self.conv1 = nn.Conv2d(c, c, kernel_size = 3, padding = 1)
        self.norm = nn.InstanceNorm2d(c)
        self.conv2 = nn.Conv2d(c, 1, kernel_size = 1)

    def forward(self, x, z_a):
        m = torch.cat([x, z_a], dim = 1)
        m = self.conv1(m)
        m = nn.LeakyReLU(0.2)(m)
        m = self.norm(m)
        m = self.conv2(m)
        m = torch.reshape(m, (-1, self.size, self.size))
        m = nn.Sigmoid()(m)
        return torch.mul(x, m) + torch.mul(z_a, 1 - m)

class AdaIN_RB(nn):
    def __init__(self, c_in, c_out, resample = "down"):
        """
        param:
        c_in is the channel of input
        c_out is the channel of output
        resample is whether down or up sample
        """
        super(AdaIN_RB, self).__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size = 3, padding = 1)
        self.res_conv = nn.Conv2d(c_in, c_out, kernel_size = 3, padding = 1)
        if resample == "down":
            scale_factor = 0.5
        elif resample == "up":
            scale_factor = 2
        else:
            scale_factor = 1
        self.resample = lambda m: F.interpolate(m, scale_factor = scale_factor)

    def forward(self, x, w_id):
        m = AdaIN()(x, w_id)
        m = nn.LeakyReLU(0.2)(m)
        m = self.conv(m)
        m = self.resample(m)

        m2 = self.res_conv(x)
        m2 = self.resample(m2)

        return m + m2

class RB(nn):
    def __init__(self, c_in, c_out, resample = "down"):
        """
        param:
        c_in is the channel of input
        c_out is the channel of output
        resample is whether down or up sample
        """
        super(RB, self).__init__()
        self.norm = nn.InstanceNorm2d(c_in)
        self.conv = nn.Conv2d(c_in, c_out, kernel_size = 3, padding = 1)
        self.res_conv = nn.Conv2d(c_in, c_out, kernel_size = 3, padding = 1)
        if resample == "down":
            scale_factor = 0.5
        elif resample == "up":
            scale_factor = 2
        else:
            scale_factor = 1
        self.resample = lambda m: F.interpolate(m, scale_factor = scale_factor)

    def forward(self, x):
        m = self.norm(x)
        m = nn.LeakyReLU(0.2)(m)
        m = self.conv(m)
        m = self.resample(m)

        m2 = self.res_conv(x)
        m2 = self.resample(m2)

        return m + m2

class AFFA_RB(nn):
    def __init__(self, c_in, c_out, s, resample = "down"):
        """
        param:
        c_in is the channel of input
        c_out is the channel of output
        s is the size of input
        resample is whether down / up sample / none
        """
        super(AFFA_RB, self).__init__()
        self.AFFA = AFFA_module(c_in, s)
        self.conv = nn.Conv2d(c_in, c_out, kernel_size = 3, padding = 1)
        self.res_conv = nn.Conv2d(c_in, c_out, kernel_size = 3, padding = 1)
        if resample == "down":
            scale_factor = 0.5
        elif resample == "up":
            scale_factor = 2
        else:
            scale_factor = 1
        self.resample = lambda m: F.interpolate(m, scale_factor = scale_factor)

    def forward(self, x, z_a, w_id):
        m = self.AFFA(x, z_a)
        m = AdaIN()(m, w_id)
        m = nn.LeakyReLU(0.2)(m)
        m = self.conv(m)
        m = self.resample(m)

        m2 = self.res_conv(x)
        m2 = self.resample(m2)

        return m + m2

class Concat_RB(nn):
    def __init__(self, c_in, c_out, resample = "down"):
        """
        param:
        c_in is the channel of input
        c_out is the channel of output
        s is the size of input
        resample is whether down or up sample
        """
        super(AFFA_RB, self).__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size = 3, padding = 1)
        self.res_conv = nn.Conv2d(c_in, c_out, kernel_size = 3, padding = 1)
        if resample == "down":
            scale_factor = 0.5
        elif resample == "up":
            scale_factor = 2
        else:
            scale_factor = 1
        self.resample = lambda m: F.interpolate(m, scale_factor = scale_factor)

    def forward(self, x, z_a, w_id):
        m = torch.cat([x, z_a], 1)
        m = AdaIN()(m, w_id)
        m = nn.LeakyReLU(0.2)(m)
        m = self.conv(m)
        m = self.resample(m)

        m2 = self.res_conv(x)
        m2 = self.resample(m2)
        return m + m2
    
class RDB(nn):
    def __init__(self, c_in, c_out):
        super(RDB, self).__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size = 1, padding = "same")
        self.resample = nn.AvgPool2d(kernel_size = 2, stride = 2)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size = 3, padding = "same")

    def forward(self, x, resample = True):
        """
        resample: whether do
        """
        r = self.conv1(x)
        r = self.resample(r) if resample else r

        x = nn.InstanceNorm2d(x.shape[1])(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv2(x)
        x = self.resample(x) if resample else x

        return x + r
    
