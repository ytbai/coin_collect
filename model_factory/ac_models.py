import torch
import numpy as np
import pandas as pd
import os
from torch import nn

from model_factory.modules import *
from data_factory import *


class ACBaseline(nn.Module):
  def __init__(self):
    super().__init__()
    
    self.trunk = nn.Sequential(
        nn.Conv2d(Game.num_channels, 8, kernel_size = 3, padding = 1),
        nn.ReLU(),

        ResidualBlock(8),
        ResidualDownsample(8),
        ResidualBlock(16),

        GlobalAvgPool2d(),
        nn.BatchNorm1d(16),
        nn.ReLU(),
        )

    self.actor_head = nn.Sequential(
        nn.Linear(16, Game.num_actions),
        Rescale(1/np.sqrt(16)),
        nn.Softmax(dim = 1),
        )
    
    self.critic_head = nn.Sequential(
        nn.Linear(16, 1),
        Rescale(1/np.sqrt(16)),
        View(shape=(-1,)),
    )

  def forward(self, x):
    base = self.trunk(x)
    actor_output = self.actor_head(base)
    critic_output = self.critic_head(base)
    return actor_output, critic_output

  def actor(self, x):
    base = self.trunk(x)
    return self.actor_head(base)
  
  def critic(self, x):
    base = self.trunk(x)
    return self.critic_head(base)



class ACBaselineWide(nn.Module):
  def __init__(self):
    super().__init__()
    
    self.trunk = nn.Sequential(
        nn.Conv2d(Game.num_channels, 16, kernel_size = 3, padding = 1),
        nn.ReLU(),

        ResidualBlock(16),
        ResidualDownsample(16),
        ResidualBlock(32),

        GlobalAvgPool2d(),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        )

    self.actor_head = nn.Sequential(
        nn.Linear(32, Game.num_actions),
        Rescale(1/np.sqrt(32)),
        nn.Softmax(dim = 1),
        )
    
    self.critic_head = nn.Sequential(
        nn.Linear(32, 1),
        Rescale(1/np.sqrt(32)),
        View(shape=(-1,)),
    )

  def forward(self, x):
    base = self.trunk(x)
    actor_output = self.actor_head(base)
    critic_output = self.critic_head(base)
    return actor_output, critic_output

  def actor(self, x):
    base = self.trunk(x)
    return self.actor_head(base)
  
  def critic(self, x):
    base = self.trunk(x)
    return self.critic_head(base)