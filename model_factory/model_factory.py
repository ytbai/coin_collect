import torch
import numpy as np
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import pickle

class ModelFactory():
  def __init__(self, model, model_dir, name = None):
    self.model = model
    self.model_dir = model_dir

    if name is None:
      self.name = self.model.__class__.__name__
    else:
      self.name = name

    self.state_dict_path = os.path.join(self.model_dir, "state_dict.tar")
    self.loss_dict_path = os.path.join(self.model_dir, "loss_dict.p")
    self.loss_dict = defaultdict(list)

  def save_best(self, loss_name, mode):
    if mode == "min":
      if min(self.loss_dict[loss_name]) == self.loss_dict[loss_name][-1]:
        self.save_state_dict()
    elif mode == "max":
      if max(self.loss_dict[loss_name]) == self.loss_dict[loss_name][-1]:
        self.save_state_dict()
    self.save_loss_dict()

  def save(self):
    self.save_state_dict()
    self.save_loss_dict()
  
  def load(self):
    self.load_state_dict()
    self.load_loss_dict()

  def save_state_dict(self):
    torch.save(self.model.state_dict(), self.state_dict_path)
    print("state_dict saved")
  
  def load_state_dict(self):
    self.model.load_state_dict(torch.load(self.state_dict_path))

  def save_loss_dict(self):
    f = open(self.loss_dict_path, "wb")
    pickle.dump(self.loss_dict, f)
    f.close()
    print("loss_dict saved")
    
  def load_loss_dict(self):
    f = open(self.loss_dict_path, "rb")
    self.loss_dict = pickle.load(f)
    f.close()

  def add_loss_name(self, loss_name):
    self.loss_dict[loss_name] = []

  def append_loss(self, loss_name, loss_val):
    self.loss_dict[loss_name].append(loss_val)

  def plot_loss_dict(self, include_model_name = False):
    for loss_name in self.loss_dict:
      if include_model_name:
        label = self.name + " " + loss_name
      else:
        label = loss_name
      plt.plot(self.loss_dict[loss_name], label = label)
    plt.legend()

  def get_last_epoch(self):
    return epochs

  def print_last_loss(self, epoch = None, verbose = True):
    if verbose == False:
      return
    if epoch is not None:
      print("epoch: %d" % epoch, end = " ")

    for key in self.loss_dict:
      print("%s %f" % (key, self.loss_dict[key][-1]), end = " ")

    print("")