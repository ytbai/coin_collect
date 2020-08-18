import torch
import numpy as np
import gc
from data_factory.game import *
from data_factory.sars import *

class QSimulator():
  def __init__(self, Nx, Ny, Q):
    self.Nx = Nx
    self.Ny = Ny
    self.Q = Q
    self.dataset = None
    self.renew_dataset()
    self.T = 64

  def renew_dataset(self):
    if self.dataset is not None:
      del self.dataset
      gc.collect()
    
    self.dataset = SARSDataset(self.Nx, self.Ny)
    return self

  def simulate(self, N, eps):
    for _ in range(N):
      self.simulate_once(eps)
    return self

  def simulate_once(self, eps):
    game = Game(self.Nx, self.Ny)
    while game.t < self.T and not game.terminal():
      S = game.get_state()
      A = self.eval_eps_greedy_action(S, eps)
      Sp, R = game.advance(A)
      self.dataset.append(S, A, R, Sp)
    return self
  
  def eval_eps_greedy_action(self, S, eps):
    self.Q.eval()
    e = np.random.uniform()
    if e > eps:
      max_action = self.eval_max_action(S)
      return max_action
    else:
      explore_action = np.random.randint(9)
      return explore_action

  def eval_max_action(self, S):
    self.Q.eval()
    Q_values = self.Q(S.view(1, 4, self.Nx, self.Ny))
    max_action = torch.argmax(Q_values).item()
    return max_action
  
  def terminal(self, S):
    return 1-torch.min(torch.sum(S[:,0,:,:], dim = (1,2)), torch.tensor(1).type(torch.cuda.FloatTensor))

  def get_dataloader(self, batch_size = 32):
    return torch.utils.data.DataLoader(self.dataset, batch_size = batch_size, shuffle = True, drop_last = True)
