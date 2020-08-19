import torch
import numpy as np
import pandas as pd
import os
from torch import nn

class Coin():
  def __init__(self, Nx, Ny, x = None, y = None, vx = None, vy = None):
    self.Nx = Nx
    self.Ny = Ny
    if x is None:
      self.x = np.random.randint(Nx)
    else:
      self.x = x
    
    if y is None:
      self.y = np.random.randint(Ny)
    else:
      self.y = y

    if vx is None:
      self.vx = np.random.randint(-1,2)
    else:
      self.vx = vx
    
    if vy is None:
      self.vy = np.random.randint(-1,2)
    else:
      self.vy = vy
  
  def reflect_x(self):
    return self.x + self.vx < 0 or self.x + self.vx >= self.Nx

  def reflect_y(self):
    return self.y + self.vy < 0 or self.y + self.vy >= self.Ny

  def advance(self):
    if self.reflect_x():
      vx = -self.vx
    else:
      vx = self.vx
    
    if self.reflect_y():
      vy = -self.vy
    else:
      vy = self.vy
    
    x = self.x + vx
    y = self.y + vy

    return Coin(self.Nx, self.Ny, x, y, vx, vy)
  
  def print(self):
    print("Coin | pos (%d, %d) | vel (%d, %d)" % (self.x, self.y, self.vx, self.vy))


class Agent():
  def __init__(self, Nx, Ny, x = None, y = None):
    self.Nx = Nx
    self.Ny = Ny
    if x is None:
      self.x = np.random.randint(self.Nx)
    else:
      self.x = x
    
    if y is None:
      self.y = np.random.randint(self.Ny)
    else:
      self.y = y
  
  def can_collect(self, coin):
    if self.x == coin.x and self.y == coin.y:
      return True
    else:
      return False

  def advance(self, action):
    vx, vy = Action.unflatten(action)
    x = (self.x + vx) % self.Nx
    y = (self.y + vy) % self.Ny
    return Agent(self.Nx, self.Ny, x, y)
  
  def print(self):
    print("Agent | pos (%d, %d)" % (self.x, self.y))


class Action():
  @staticmethod
  def unflatten(action):
    vx = action // 3
    vy = action - 3*vx

    vx -= 1
    vy -= 1
    return vx, vy

  @staticmethod
  def flatten(vx, vy):
    vx += 1
    vy += 1
    action = 3*vx + vy
    return action

  @staticmethod
  def collect(action):
    return Action.unflatten(action) == (0,0)

class Board():
  def __init__(self, Nx, Ny):
    self.Nx = Nx
    self.Ny = Ny
    self.state = torch.zeros(Game.num_channels, self.Nx, self.Ny, requires_grad = False).type(torch.cuda.FloatTensor)
    self.coin_list = []
    self.agent = None
    
  def add_coin(self, coin = None):
    if coin is None:
      coin = Coin(self.Nx, self.Ny)

    self.coin_list.append(coin)
    self.state[0][coin.x][coin.y] += 1
    self.state[1][coin.x][coin.y] += coin.vx
    self.state[2][coin.x][coin.y] += coin.vy
    return self
  
  def add_coins(self, coins = None):
    if isinstance(coins, int):
      num_coins = coins
      for _ in range(num_coins):
        self.add_coin()
    elif isinstance(coins, list):
      for coin in coins:
        self.add_coin(coin)
    return self


  def add_agent(self, agent = None):
    if agent is None:
      agent = Agent(self.Nx, self.Ny)

    self.agent = agent
    self.state[3][agent.x][agent.y] += 1
    return self
  
  def terminal(self):
    return len(self.coin_list) == 0

  def advance(self, action):
    new_board = Board(self.Nx, self.Ny)

    num_coins_collected = 0

    for coin in self.coin_list:
      if Action.collect(action) and self.agent.can_collect(coin):
        num_coins_collected += 1
      else:
        new_board.add_coin(coin.advance())
    
    new_board.add_agent(self.agent.advance(action))
    return new_board, num_coins_collected

  def print(self):
    for coin in self.coin_list:
      coin.print()
    self.agent.print()
    print(self.state[0] +0.5*self.state[3])


  def get_state(self):
    return self.state

class Game():
  num_channels = 4
  num_actions = 9

  @staticmethod
  def state_to_batch(state):
    return state.view(1, *tuple(state.shape))

  @staticmethod
  def board_to_batch(board):
    return Game.state_to_batch(board.get_state())

  @staticmethod
  def prob_to_action(prob):
    prob = torch.flatten(prob).detach().cpu().numpy()
    return np.random.choice(Game.num_actions, p = prob)

  @staticmethod
  def project(X, A):
    A_one_hot = nn.functional.one_hot(A, num_classes = Game.num_actions).type(torch.cuda.FloatTensor)
    projection = torch.sum(X * A_one_hot, dim = 1)
    return projection

  def __init__(self, Nx, Ny, lambda_coins = 3):
    self.lambda_coins = lambda_coins

    self.Nx = Nx
    self.Ny = Ny
    self.board_history = []
    self.reward_history = []
    self.action_history = []
    self.init_board_poisson()


    

  def init_board_poisson(self):
    board = Board(self.Nx, self.Ny)
    num_coins = np.random.poisson(lam = self.lambda_coins)
    board.add_coins(num_coins)
    board.add_agent()
    self.board_history.append(board)
    self.t = 0
    return self

  def terminal(self):
    return self.get_board().terminal()

  def advance(self, action):
    if self.terminal():
      return None
    
    new_board, num_coins_collected = self.get_board().advance(action)
    new_reward = torch.tensor(-1 + num_coins_collected).type(torch.cuda.FloatTensor)

    self.add_board(new_board)
    self.add_reward(new_reward)
    self.add_action(action)
    self.t += 1

    new_state = new_board.get_state()
    return new_state, new_reward

  def add_board(self, board):
    self.board_history.append(board)
  
  def add_reward(self, reward):
    self.reward_history.append(reward)

  def add_action(self, action):
    self.action_history.append(action)

  def get_reward(self, t = -1):
    return self.reward_history[t]

  def get_board(self, t = -1):
    return self.board_history[t]

  def get_state(self, t = -1):
    return self.board_history[t].state

  def get_total_reward(self):
    return sum(self.reward_history)

  def print_board(self, t = -1):
    self.get_board(t).print()