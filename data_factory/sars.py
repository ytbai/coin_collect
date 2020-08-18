import torch
import numpy as np

class SARSDataset(torch.utils.data.Dataset):
  def __init__(self, Nx, Ny):
    self.Nx = Nx
    self.Ny = Ny
    self.state_list = []
    self.action_list = []
    self.reward_list = []
    self.state_p_list = []
    self.term_list = []
    self.t_list = []
  
  def append(self, state, action, reward, state_p, term, t):
    self.state_list.append(state)
    self.action_list.append(action)
    self.reward_list.append(reward)
    self.state_p_list.append(state_p)
    self.term_list.append(term)
    self.t_list.append(t)

    return self

  def __getitem__(self, index):
    state = self.state_list[index]
    action = self.action_list[index]
    reward = self.reward_list[index]
    state_p = self.state_p_list[index]
    term = self.term_list[index]
    t = self.t_list[index]
    return state, action, reward, state_p, term, t
  
  def __len__(self):
    return len(self.state_list)