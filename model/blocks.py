import torch
import torch.nn.functional as F
from torch import nn
from AdaIN import AdaIN

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

class AFFA_module(nn):
    def __init__(self, c, s):
        """
        :param c: the channel of input
        :param s: the size of input (equals to W or H)
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
    def __init__(self, c_in, c_out, lantent, resample = "down"):
        """
        :param c_in: the channel of input
        :param c_out: the channel of output
        :param resample: down / up sample / none
        """
        super(AdaIN_RB, self).__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size = 3, padding = 1)
        self.AdaIN = AdaIN(lantent, c_in)
        self.res_conv = nn.Conv2d(c_in, c_out, kernel_size = 3, padding = 1)
        if resample == "down":
            scale_factor = 0.5
        elif resample == "up":
            scale_factor = 2
        else:
            scale_factor = 1
        self.resample = lambda m: F.interpolate(m, scale_factor = scale_factor)

    def forward(self, x, w_id):
        m = self.AdaIN(x, w_id)
        m = nn.LeakyReLU(0.2)(m)
        m = self.conv(m)
        m = self.resample(m)

        m2 = self.res_conv(x)
        m2 = self.resample(m2)

        return m + m2

class RB(nn):
    def __init__(self, c_in, c_out, resample = "down"):
        """
        :param c_in: the channel of input
        :param c_out: the channel of output
        :param resample: down / up sample / none
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
    def __init__(self, c_in, c_out, lantent, s, resample = "down"):
        """
        :param c_in: the channel of input
        :param c_out: the channel of output
        :param s: the size of input
        :param resample: down / up sample / none
        """
        super(AFFA_RB, self).__init__()
        self.AFFA = AFFA_module(c_in, s)
        self.AdaIN = AdaIN(lantent, c_in)
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
        m = AdaIN(m, w_id)
        m = nn.LeakyReLU(0.2)(m)
        m = self.conv(m)
        m = self.resample(m)

        m2 = self.res_conv(x)
        m2 = self.resample(m2)

        return m + m2

class Concat_RB(nn):
    def __init__(self, c_in, c_out, lantent, resample = "down"):
        """
        :param c_in: the channel of input
        :param c_out: the channel of output
        :param s: the size of input
        :param resample: down / up sample / none
        """
        super(AFFA_RB, self).__init__()
        self.AdaIN = AdaIN(lantent, c_in)
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
        m = AdaIN(m, w_id)
        m = nn.LeakyReLU(0.2)(m)
        m = self.conv(m)
        m = self.resample(m)

        m2 = self.res_conv(x)
        m2 = self.resample(m2)
        return m + m2

class RDB(nn):
    def __init__(self, c_in, c_out):
        """
        :param c_in: the channel of input
        :param c_out: the channel of output
        """
        super(RDB, self).__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size = 1, padding = "same")
        self.resample = nn.AvgPool2d(kernel_size = 2, stride = 2)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size = 3, padding = "same")

    def forward(self, x, resample = True):
        """
        :param resample: whether apply resampling
        """
        r = self.conv1(x)
        r = self.resample(r) if resample else r

        x = nn.InstanceNorm2d(x.shape[1])(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv2(x)
        x = self.resample(x) if resample else x

        return x + r
    
