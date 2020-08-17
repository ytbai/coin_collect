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
        nn.BatchNorm2d(8),
        nn.ReLU(),

        nn.Conv2d(8, 8, kernel_size = 3, padding = 1),
        nn.BatchNorm2d(8),
        nn.ReLU(),

        nn.Conv2d(8, 16, kernel_size = 3, stride = 2, padding = 1),
        nn.BatchNorm2d(16),
        nn.ReLU(),

        nn.Conv2d(16, 16, kernel_size = 3, padding = 1),
        nn.BatchNorm2d(16),
        nn.ReLU(),

        nn.Conv2d(16, 16, kernel_size = 3, padding = 1),
        nn.BatchNorm2d(16),
        nn.ReLU(),

        nn.Conv2d(16, 32, kernel_size = 3, stride = 2, padding = 1),
        nn.BatchNorm2d(32),
        nn.ReLU(),

        nn.Conv2d(32, 32, kernel_size = 3, padding = 1),
        nn.BatchNorm2d(32),
        nn.ReLU(),

        nn.Conv2d(32, 32, kernel_size = 3, padding = 1),
        nn.BatchNorm2d(32),
        nn.ReLU(),

        GlobalAvgPool2d(),
        nn.BatchNorm1d(32),
        nn.ReLU(),

        nn.Linear(32, 9),
    )


  def forward(self, x):
    return self.seq(x)
