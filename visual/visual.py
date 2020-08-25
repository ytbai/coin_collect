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

  def plot_history(self, width = 2):
    fig, ax = plt.subplots(1,self.game.t+1, figsize = (width*(self.game.t+1), width))
    for t in range(self.game.t + 1):
      combined_state = self.get_combined_state(t)
      if self.game.t == 0:
        ax_t = ax
      else:
        ax_t = ax[t]
      ax_t.imshow(combined_state)
      ax_t.axis("off")
      ax_t.set_title("%s" % (self.get_action_str(t)))
    return fig, ax