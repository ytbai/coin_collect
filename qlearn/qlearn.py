import torch
import numpy as np

from model_factory import *
from data_factory import *
from test import *

class QLearn():
  def __init__(self, name, Nx, Ny, N_valid = 2048, lr_init = 1e-4, kappa = None):
    self.Nx = Nx
    self.Ny = Ny
    self.Q = QValue().cuda()
    self.name = name
    self.N_valid = N_valid
    self.lr_init = lr_init
    self.kappa = kappa
    self.model_dir = os.path.join("qlearn/models", self.name)
    self.model_factory = ModelFactory(model = self.Q, model_dir = self.model_dir, name = self.name)

    self.simulator = QSimulator(self.Nx, self.Ny, self.Q)
    self.valid_simulator = TestSimulator(self.Nx, self.Ny)
    
    self.init_optim()
    

  def init_optim(self):
    self.optimizer = torch.optim.Adam(params = self.Q.parameters(), lr = self.lr_init)
    self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="max", factor = 0.1, patience = 10, verbose = True)
    self.criterion = nn.MSELoss()

  def R_process(self, R):
    return R / self.simulator.T

  def get_Q_target(self, R, Sp, term):
    self.Q.eval()
    Q_max = torch.max(self.Q(Sp).detach(), dim = 1)[0]
    Q_max *= 1-term
    R_proc = self.R_process(R)
    Q_target = R_proc + Q_max
    return Q_target

  def get_Q_pred(self, S, A):
    self.Q.train()
    A_one_hot = nn.functional.one_hot(A, num_classes = 9).type(torch.cuda.FloatTensor)
    Q_pred = torch.sum(self.Q(S) * A_one_hot, dim = 1)
    return Q_pred

  def train_once(self, iterations, batch_size):
    dataloader = self.simulator.get_dataloader(batch_size)
    loss_total = 0

    for it in range(iterations):
      for S, A, R, Sp, term, t in dataloader:
        self.optimizer.zero_grad()
        Q_target = self.get_Q_target(R, Sp, term)
        Q_pred = self.get_Q_pred(S, A)
        
        if self.kappa is None:
          loss = self.criterion(Q_pred, Q_target)
        else:
          loss = (Q_pred-Q_target)**2
          loss *= torch.pow(self.kappa, self.simulator.T - t)
          loss = torch.mean(loss)

        loss.backward()
        self.optimizer.step()

        loss_total += loss.item() * S.shape[0]
    loss_mean = loss_total / len(self.simulator.dataset) / iterations
    self.model_factory.append_loss("loss_train", loss_mean)
    return loss_mean
  
  def train(self, epochs, eps, N = 128, iterations = 8, batch_size = 64, verbose = False):
    for e in range(epochs):
      self.simulator.renew_dataset()
      self.simulator.simulate(N = N, eps = eps)
      self.train_once(iterations, batch_size)
      self.model_factory.print_last_loss(e, verbose)
  
  def valid(self, renew_dataset = True, save_best = True):
    if renew_dataset:
      self.valid_simulator.init_games(self.N_valid)
    loss_valid = self.valid_simulator.test(self)
    self.model_factory.append_loss("loss_valid", loss_valid)
    self.scheduler.step(loss_valid)

    if save_best:
      self.model_factory.save_best("loss_valid", mode = "max")
  
  def __call__(self, state):
    self.Q.eval()
    S = state.view(1, 4, self.Nx, self.Ny)
    return torch.argmax(self.Q(S).detach()).item()

