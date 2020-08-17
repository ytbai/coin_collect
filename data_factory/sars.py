import torch
import numpy as np

class SARSDataset(torch.utils.data.Dataset):
  def __init__(self, Nx = 8, Ny = 8):
    self.Nx = Nx
    self.Ny = Ny
    self.S_list = []
    self.A_list = []
    self.R_list = []
    self.Sp_list = []
  
  def append(self, S, A, R, Sp):
    self.S_list.append(S)
    self.A_list.append(A)
    self.R_list.append(R)
    self.Sp_list.append(Sp)

    return self

  def __getitem__(self, index):
    S = self.S_list[index]
    A = self.A_list[index]
    R = self.R_list[index]
    Sp = self.Sp_list[index]
    return S, A, R, Sp
  
  def __len__(self):
    return len(self.S_list)