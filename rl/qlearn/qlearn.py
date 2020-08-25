import torch
import numpy as np

from model_factory import *
from data_factory import *
from rl import *
from test import *

class QLearn(RL):
  def __init__(self, Nx, Ny, name, Q_class, eps = 0.1, N_valid = 1024, lr_init = 1e-4, kappa = None):
    super().__init__(Nx, Ny, name, Q_class, models_dir = "rl/qlearn/models", N_valid = N_valid, lr_init = lr_init)

    self.Q = self.model
    self.eps = eps
    self.kappa = kappa

    self.simulator = QSimulator(self.Nx, self.Ny, self.Q, self.eps)

  def get_Q_target(self, R, Sp, term):
    self.Q.eval()
    Q_max = torch.max(self.Q(Sp).detach(), dim = 1)[0]
    Q_max *= 1-term
    R_proc = self.R_process(R)
    Q_target = R_proc + Q_max
    return Q_target

  def get_Q_pred(self, S, A):
    self.Q.train()
    Q_pred = Game.project(self.Q(S), A)
    return Q_pred

  def train_once(self, N, iterations, batch_size):
    self.simulator.renew_dataset()
    self.simulator.simulate(N = N)
    dataloader = self.simulator.get_dataloader(batch_size)
    loss_total = 0

    for it in range(iterations):
      for S, A, R, Sp, term, t in dataloader:
        self.optimizer.zero_grad()
        Q_target = self.get_Q_target(R, Sp, term)
        Q_pred = self.get_Q_pred(S, A)
        
        if self.kappa is None:
          loss = nn.MSELoss()(Q_pred, Q_target)
        else:
          reduce = torch.pow(self.kappa, self.simulator.T - t)
          loss = nn.MSELoss()(Q_pred * reduce, Q_target * reduce)

        loss.backward()
        self.optimizer.step()

        loss_total += loss.item() * S.shape[0]
    loss_mean = loss_total / len(self.simulator.dataset) / iterations
    self.model_factory.append_loss("loss_train", loss_mean)
    return loss_mean
  

