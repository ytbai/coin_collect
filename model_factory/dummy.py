import torch
import numpy as np
from torch import nn

from data_factory import *

class DummyModel(nn.Module):
  def __init__(self, mode):
    super().__init__()
    self.mode = mode

  def forward(self, x):
    if self.mode == "random":
      return np.random.randint(Game.num_actions)
    elif self.mode == "collect":
      return Action.get_action("collect")