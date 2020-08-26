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

  def test_once(self, model_wrapper, game):
    model_wrapper.model.eval()
    game = copy.deepcopy(game)
    while game.t < self.T and not game.terminal():
      state = game.get_state()
      action = model_wrapper(state)
      state_p, reward = game.advance(action)
    return game
  
  def test(self, model_wrapper):
    total_return = 0
    for index in range(self.N_test):
      game = self.games[index]
      total_return += self.test_once(model_wrapper, game).get_total_reward()
    
    mean_return = total_return / self.N_test
    return mean_return


class BaselinePlayer():
  def __init__(self, mode):
    self.mode = mode

  def __call__(self, x):
    if self.mode == "random":
      return np.random.randint(Game.num_actions)
    elif self.mode == "collect":
      return Action.get_action("collect")

