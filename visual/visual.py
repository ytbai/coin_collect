import torch
import numpy as np

from data_factory import *

class Visual():
  def __init__(self, game):
    self.game = game
    self.Nx, self.Ny = self.game.Nx, self.game.Ny
    self.coin_color = (1,1,0)
    self.agent_color_move = (1,0,0)
    self.agent_color_collect = (1,0,1)

  def colorize_state(self, singlet_state, color):
    R, G, B = color
    singlet_state = np.clip(singlet_state, 0, 1)
    colored_state = np.stack([singlet_state * R, singlet_state * G,singlet_state * B], axis=-1)
    return colored_state

  def combine_colored_states(self, agent_state, coin_state):
    combined_state = np.copy(coin_state)
    for i in range(self.Nx):
      for j in range(self.Ny):
        if agent_state[i][j].sum() > 0:
          combined_state[i][j] = agent_state[i][j]
    return combined_state

  def get_agent_color(self, t):
    if t >= self.game.t:
      return self.agent_color_move
    action = self.game.action_history[t]
    if Action.collect(action):
      return self.agent_color_collect
    else:
      return self.agent_color_move

  def get_action_str(self, t):
    if t == self.game.t:
      return Action.get_terminal_str()
    action = self.game.action_history[t]
    return Action.action_to_str(action)

  def get_combined_state(self, t):
    state_np = self.game.board_history[t].get_state().cpu().numpy()
    agent_color = self.get_agent_color(t)
    agent_state = self.colorize_state(state_np[3], agent_color)
    coin_state = self.colorize_state(state_np[0], self.coin_color)
    combined_state = self.combine_colored_states(agent_state, coin_state)
    return combined_state

  def plot_history(self, width = 2, n_cols = 8):
    n_boards = self.game.t + 1
    n_rows = (n_boards-1)//n_cols + 1

    fig, ax = plt.subplots(n_rows, n_cols, figsize = (width*n_cols, width*n_rows))
    for t in range(n_rows * n_cols):
      if self.game.t == 0:
        ax_t = ax
      elif n_rows == 1:
        ax_t = ax[t]
      else:
        t_row = t // n_cols 
        t_col = t % n_cols
        ax_t = ax[t_row][t_col]
      ax_t.axis("off")

      if t < n_boards:
        combined_state = self.get_combined_state(t)
        ax_t.imshow(combined_state)
        ax_t.set_title("%s" % (self.get_action_str(t)))
      ax_t.axis("off")
      
    return fig, ax