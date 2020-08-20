import torch
import numpy as np
import gc
from data_factory import *
from model_factory import *

class Simulator():
  def __init__(self, Nx, Ny):
    self.Nx = Nx
    self.Ny = Ny
    self.dataset = None
    self.renew_dataset()
    self.T = 64  
  
  def renew_dataset(self):
    if self.dataset is not None:
      del self.dataset
      gc.collect()
    
    self.dataset = SARSDataset(self.Nx, self.Ny)
    return self

  def simulate(self, N):
    for _ in range(N):
      self.simulate_once()
    return self

  def simulate_once(self):
    game = Game(self.Nx, self.Ny)
    while game.t < self.T and not game.terminal():
      state = game.get_state()
      action = self.eval_action(state)
      state_prime, reward = game.advance(action)
      term = torch.tensor(int(game.terminal())).type(torch.cuda.FloatTensor)
      t = torch.tensor(game.t).type(torch.cuda.FloatTensor)
      self.dataset.append(state, action, reward, state_prime, term, t)
    return self

  def get_dataloader(self, batch_size = 32):
    return torch.utils.data.DataLoader(self.dataset, batch_size = batch_size, shuffle = True, drop_last = True)


class QSimulator(Simulator):
  def __init__(self, Nx, Ny, Q, eps):
    super().__init__(Nx, Ny)
    self.Q = Q
    self.eps = eps

  def eval_action(self, state):
    return self.eval_eps_greedy_action(state)
  
  def eval_eps_greedy_action(self, state):
    self.Q.eval()
    e = np.random.uniform()
    if e > self.eps:
      max_action = self.eval_max_action(state)
      return max_action
    else:
      explore_action = np.random.randint(Game.num_actions)
      return explore_action

  def eval_max_action(self, state):
    self.Q.eval()
    S = Game.state_to_batch(state)
    Q_values = self.Q(S)
    max_action = torch.argmax(Q_values).item()
    return max_action

  def __call__(self, state):
    return self.eval_max_action(state)


class ActorSimulator(Simulator):
  def __init__(self, Nx, Ny, actor_wrapper):
    super().__init__(Nx, Ny)
    self.actor_wrapper = actor_wrapper
  
  def eval_action(self, state):
    self.actor_wrapper.eval()
    S = Game.state_to_batch(state)
    actor_values = self.actor_wrapper.actor(S)
    action = Game.prob_to_action(actor_values)
    return action

  def __call__(self, state):
    return self.eval_action(state)
