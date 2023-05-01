import torch
from torch import nn

class mapping(nn):
    def __init__(self, dim = 512):
        super(mapping, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.fc3 = nn.Linear(dim, dim)
        self.fc4 = nn.Linear(dim, 256)

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
    def __init__(self, in_c, out_c):
        super(AFFA_module, self).__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size = 3, padding = 1)
        self.norm = nn.InstanceNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, 1, kernel_size = 1)

    def forward(self, x, z_a):
        m = torch.cat([x, z_a], dim = 1)
        m = self.conv1(m)
        m = nn.LeakyReLU(0.2)(m)
        m = self.norm(m)
        m = self.conv2(m)
        m = nn.Sigmoid()(m)
        return torch.mul(x, m) + torch.mul(z_a, 1 - m)

class AFFA_RB(nn):
    def __init__(self):
        super(AFFA_RB, self).__init__()

    def forward(self, x, z_a, w_id):
        pass 

class AdaIN_RB(nn):
    def __init__(self):
        super(AdaIN_RB, self).__init__()

    def forward(self, x, w_id):
        pass

class RB(nn):
    def __init__(self):
        super(RB, self).__init__()

    def forward(self, x):
        pass