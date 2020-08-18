import torch
import numpy as np
import pandas as pd
import os
from torch import nn


class GlobalAvgPool2d(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x):
    return torch.mean(x, dim=(2,3))


class ResidualBlock(nn.Module):
  def __init__(self, in_channels):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = in_channels

    self.pre_res = nn.Sequential(
                  nn.Conv2d(self.in_channels, self.in_channels, kernel_size = 3, padding = 1),
                  nn.BatchNorm2d(self.in_channels),
                  nn.ReLU(),
                  nn.Conv2d(self.in_channels, self.in_channels, kernel_size = 3, padding = 1),
                  )
    self.post_res = nn.Sequential(
                  nn.BatchNorm2d(self.in_channels),
                  nn.ReLU(),
                  )

  def forward(self, x):
    output = self.pre_res(x)
    output = output + x
    output = self.post_res(x)
    return output

class ResidualDownsample(nn.Module):
  def __init__(self, in_channels):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = 2*in_channels

    self.pre_res = nn.Sequential(
                  nn.Conv2d(self.in_channels, self.out_channels, kernel_size = 3, stride = 2, padding = 1),
                  )
    self.res_reduce = nn.Sequential(
                  nn.AvgPool2d(kernel_size = 2, stride = 2),
                  nn.Conv2d(self.in_channels, self.out_channels, kernel_size = 1),
                  )
    self.post_res = nn.Sequential(
                  nn.BatchNorm2d(self.out_channels),
                  nn.ReLU(),
                  )

  def forward(self, x):
    output = self.pre_res(x)
    output = output + self.res_reduce(x)
    output = self.post_res(output)
    return output

