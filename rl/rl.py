import torch
import numpy as np
import os

from model_factory import *
from test import *

class RL():
  def __init__(self, Nx, Ny, name, model_class, models_dir, N_valid, lr_init = 1e-4):
    self.Nx = Nx
    self.Ny = Ny
    self.name = name
    self.model = model_class().cuda()
    
    self.model_dir = os.path.join(models_dir, self.name)
    self.model_factory = ModelFactory(self.model, model_dir = self.model_dir, name = self.name)
    self.lr_init = lr_init
    self.valid_best = "max"
    self.init_optim()
    self.valid_simulator = TestSimulator(self.Nx, self.Ny)
    self.N_valid = N_valid

  def R_process(self, R):
    return R/self.simulator.T

  def init_optim(self):
    self.optimizer = torch.optim.Adam(params = self.model.parameters(), lr = self.lr_init)
    self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode=self.valid_best, factor = 0.1, patience = 10, verbose = True)

  def train(self, epochs, N = 64, iterations = 4, batch_size = 64, verbose = False):
    for e in range(epochs):
      self.simulator.renew_dataset()
      self.simulator.simulate(N = N)
      self.train_once(iterations, batch_size)
      self.model_factory.print_last_loss(e, verbose)

  
  def valid(self, renew_dataset = True, save_best = True):
    if renew_dataset:
      self.valid_simulator.init_games(self.N_valid)
    score_valid = self.valid_simulator.test(self)
    self.model_factory.append_loss("score_valid", score_valid)
    self.scheduler.step(score_valid)

    if save_best:
      self.model_factory.save_best("score_valid", mode = self.valid_best)

  def __call__(self, state):
    return self.simulator(state)