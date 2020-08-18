import torch
import numpy as np
import copy

from data_factory import *

class TestSimulator():
  def __init__(self, Nx, Ny):
    self.Nx = Nx
    self.Ny = Ny
    self.T = 64
    self.games = []
    self.N_test = 0
  
  def init_games(self, N_test):
    self.games = []
    for _ in range(N_test):
      game = Game(self.Nx, self.Ny)
      self.games.append(game)
    self.N_test = N_test
    print("%d test games initialized" % N_test)
    return self

  def test_once(self, model_wrapper, index):
    total_reward = 0
    game = copy.deepcopy(self.games[index])
    while game.t < self.T and not game.terminal():
      state = game.get_state()
      action = model_wrapper(state)
      state_p, reward = game.advance(action)
      total_reward += reward
    return total_reward
  
  def test(self, model_wrapper):
    total_return = 0
    for index in range(self.N_test):

      total_return += self.test_once(model_wrapper, index)
    
    mean_return = total_return / self.N_test
    return mean_return


class RandomPlayer():
  def __init__(self):
    return

  def __call__(self, x):
    return np.random.randint(9)

