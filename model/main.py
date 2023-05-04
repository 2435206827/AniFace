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
        self.RB6 = RB(512, 512, resample = "none")

        # Decoder
        self.AdaIN_RB = AdaIN_RB(512, 512, "none")
        self.AFFA_RB5 = AFFA_RB(512, 512, 2, "up")
        self.AFFA_RB4 = AFFA_RB(512, 512, 4, "up")
        self.AFFA_RB3 = AFFA_RB(512, 256, 8, "up")
        self.AFFA_RB2 = AFFA_RB(256, 128, 16, "up")
        self.AFFA_RB1 = AFFA_RB(128, 64, 32, "up")
        self.Concat_RB = Concat_RB(64, 3, "none")

    def forward(self, w_id, x_s):
        x_0 = self.conv1(x_s)
        x_1 = self.RB1(x_0)
        x_2 = self.RB2(x_1)
        x_3 = self.RB3(x_2)
        x_4 = self.RB4(x_3)
        x_5 = self.RB5(x_4)
        x_6 = self.RB6(x_5)

        y_6 = self.AdaIN_RB(x_6, w_id)
        y_5 = self.AFFA_RB5(y_6, x_5, w_id)
        y_4 = self.AFFA_RB5(y_5, x_4, w_id)
        y_3 = self.AFFA_RB5(y_4, x_3, w_id)
        y_2 = self.AFFA_RB5(y_3, x_2, w_id)
        y_1 = self.AFFA_RB5(y_2, x_1, w_id)
        y_0 = self.Concat_RB(y_1, x_0, w_id)
        return y_0

class Generator(nn):
    def __init__(self, arcface):
        super(Generator, self).__init__()
        self.Enc_Dec = Enc_Dec()
        self.map = mapping(1024)
        self.arcface = arcface

    def forward(self, x_t, x_s):
        """
        x_t: target face
        x_s: source face
        """
        # x_t: (N, M, C, H, W)
        x_t = torch.transpose(x_t, 0, 1)    # (N, M, C, H, W) -> (M, N, C, H, W)
        # x_zero: (M, N, 1024)
        x_zero = torch.zeros((x_t.shape[0], x_t.shape[1], 1024))
        for i in range(x_t.shape[0]):
            x_zero[i] = self.arcface(x_t[i])
        x_zero = x_zero.mean(0) # (M, N, 1024) -> (N, 1024)
        x_zero = self.map(x_zero)   # (N, 1024) -> (N, 256)

        return self.Enc_Dec(x_zero, x_s)

class Discrimator(nn):
    def __init__(self):
        super(Discrimator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, padding = "same")
        self.RDB1 = RDB(64, 128)
        self.RDB2 = RDB(128, 256)
        self.RDB3 = RDB(256, 512)
        self.RDB4 = RDB(512, 512)
        self.RDB5 = RDB(512, 512)
        self.RDB6 = RDB(512, 512)
        self.conv2 = nn.Conv2d(512, 512, kernel_size = 4)
        self.conv3 = nn.Conv2d(512, 1, kernel_size = 1)

    def forward(self, x_c):
        x_c = self.conv1(x_c)
        x_c = self.RDB1(x_c)
        x_c = self.RDB2(x_c)
        x_c = self.RDB3(x_c)
        x_c = self.RDB4(x_c)
        x_c = self.RDB5(x_c)
        x_c = self.RDB6(x_c)

        x_c = self.conv2(x_c)
        x_c = nn.LeakyReLU(0.2)(x_c)

        x_c = self.conv3(x_c)
        x_c = torch.reshape(x_c, (-1, 1))
        
        return x_c