import torch
from torch import nn

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