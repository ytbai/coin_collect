import torch
import numpy as np

from rl import *

class AC(RL):
  def __init__(self, Nx, Ny, name, actor_critic_class, N_valid, lr_init = 1e-4):
    super().__init__(Nx, Ny, name, actor_critic_class, models_dir = "rl/ac/models", N_valid = N_valid, lr_init = lr_init)

    self.actor_critic = self.model
    self.simulator = ACSimulator(self.Nx, self.Ny, self.actor_critic)
    self.lambda_critic = 0.1

  def get_delta_and_critic_target(self, S, R, Sp, term):
    self.actor_critic.eval()
    R_proc = self.R_process(R)
    critic_target = (R_proc + (1-term)*self.actor_critic.critic(Sp)).detach()
    delta = critic_target - self.actor_critic.critic(S).detach()
    return delta, critic_target

  def get_loss_actor(self, actor_output, A, delta):
    loss_actor = Game.project(actor_output, A)
    loss_actor = torch.log(loss_actor)
    loss_actor *= delta
    loss_actor = -torch.mean(loss_actor) # minus sign
    return loss_actor

  def get_loss_critic(self, critic_output, critic_target):
    loss_critic = nn.MSELoss()(critic_output, critic_target)
    return loss_critic


  def train_once(self, iterations, batch_size):
    dataloader = self.simulator.get_dataloader(batch_size)
    loss_total = 0
    loss_actor_total = 0
    loss_critic_total = 0

    for it in range(iterations):
      for S, A, R, Sp, term, t in dataloader:
        self.optimizer.zero_grad()

        delta, critic_target = self.get_delta_and_critic_target(S, R, Sp, term)

        self.actor_critic.train()
        actor_output, critic_output = self.actor_critic(S)

        loss_actor = self.get_loss_actor(actor_output, A, delta)
        loss_critic = self.get_loss_critic(critic_output, critic_target)
        loss = loss_actor + self.lambda_critic * loss_critic

        loss.backward()
        self.optimizer.step()

        loss_actor_total += loss_actor.item() * S.shape[0]
        loss_critic_total += loss_critic.item() * S.shape[0]
        loss_total += loss.item() * S.shape[0]

    loss_actor_mean = loss_actor_total / len(self.simulator.dataset) / iterations
    loss_critic_mean = loss_critic_total / len(self.simulator.dataset) / iterations
    loss_mean = loss_total / len(self.simulator.dataset) / iterations
    self.model_factory.append_loss("loss_actor_train", loss_actor_mean)
    self.model_factory.append_loss("loss_critic_train", loss_critic_mean)
    self.model_factory.append_loss("loss_train", loss_mean)
    return loss_mean