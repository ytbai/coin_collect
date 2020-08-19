import torch
import numpy as np
import pandas as pd
import os
from torch import nn

from model_factory.modules import *

class QValue(nn.Module):
  def __init__(self):
    super().__init__()
    
    self.seq = nn.Sequential(
        nn.Conv2d(4, 8, kernel_size = 3, padding = 1),
        nn.ReLU(),

        ResidualBlock(8),
        ResidualDownsample(8),
        ResidualBlock(16),

        GlobalAvgPool2d(),
        nn.BatchNorm1d(16),
        nn.ReLU(),

        nn.Linear(16, 9),
        Rescale(1/np.sqrt(16)),
    )


  def forward(self, x):
    return self.seq(x)


class QValueWide(nn.Module):
  def __init__(self):
    super().__init__()
    
    self.seq = nn.Sequential(
        nn.Conv2d(4, 16, kernel_size = 3, padding = 1),
        nn.ReLU(),

        ResidualBlock(16),
        ResidualDownsample(16),
        ResidualBlock(32),

        GlobalAvgPool2d(),
        nn.BatchNorm1d(32),
        nn.ReLU(),

        nn.Linear(32, 9),
        Rescale(1/np.sqrt(32)),
    )



  def forward(self, x):
    return self.seq(x)



class QValueWideDeep(nn.Module):
  def __init__(self):
    super().__init__()
    
    self.seq = nn.Sequential(
        nn.Conv2d(4, 16, kernel_size = 3, padding = 1),
        nn.ReLU(),

        ResidualBlock(16),
        ResidualBlock(16),
        ResidualDownsample(16),
        ResidualBlock(32),
        ResidualBlock(32),

        GlobalAvgPool2d(),
        nn.BatchNorm1d(32),
        nn.ReLU(),

        nn.Linear(32, 9),
        Rescale(1/np.sqrt(32)),
    )



  def forward(self, x):
    return self.seq(x)

class QValueDeep(nn.Module):
  def __init__(self):
    super().__init__()
    
    self.seq = nn.Sequential(
        nn.Conv2d(4, 8, kernel_size = 3, padding = 1),
        nn.ReLU(),

        ResidualBlock(8),
        ResidualBlock(8),
        ResidualDownsample(8),
        ResidualBlock(16),
        ResidualBlock(16),

        GlobalAvgPool2d(),
        nn.BatchNorm1d(16),
        nn.ReLU(),

        nn.Linear(16, 9),
        Rescale(1/np.sqrt(16)),
    )



  def forward(self, x):
    return self.seq(x)