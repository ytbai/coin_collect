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