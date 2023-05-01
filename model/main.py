import torch
from blocks import *
from torch import nn

class Enc_Dec(nn):
    def __init__(self):
        """
        Input: (3, 256, 256)
        """
        super(Enc_Dec, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 3, padding = 1)
        self.RB1 = RB(64, 128)
        self.RB2 = RB(128, 256)
        self.RB3 = RB(256, 512)
        self.RB4 = RB(512, 512)
        self.RB5 = RB(512, 512)
        self.RB6 = RB(512, 512)

        # Decoder
        self.AdaIN_RB = AdaIN_RB(512, 512, "up")
        self.AFFA_RB1 = AFFA_RB(512, 512, 2, "up")
        self.AFFA_RB2 = AFFA_RB(512, 512, 4, "up")
        self.AFFA_RB3 = AFFA_RB(512, 256, 8, "up")
        self.AFFA_RB4 = AFFA_RB(256, 128, 16, "up")
        self.AFFA_RB5 = AFFA_RB(128, 64, 32, "up")
        self.Concat_RB = None

    def forward(self, x_t, x_s):
        pass

class Generator(nn):
    def __init__(self, arcface, num_x_s = 2):
        super(Generator, self).__init__()

    def forward(self, x_t, x_s):
        pass

class Discrimator(nn):
    def __init__(self, arcface, num_x_s):
        super(Discrimator, self).__init__()

    def forward(self, x_c):
        pass

class AniFace(nn):
    def __init__(self, arcface, gen, dis, num_x_s = 2):
        super(AniFace, self).__init__()

    def forward(self, x_t, x_s):
        pass