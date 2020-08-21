import torch
import numpy as np
from rl.rl import *
from data_factory import *

class PPO(RL):
  def __init__(self, Nx, Ny, name, actor_critic_class, eps, N_valid, lr_init = 1e-4):
    super().__init__(Nx, Ny, name, actor_critic_class, models_dir = "rl/ppo/models", N_valid = N_valid, lr_init = lr_init)

    self.actor_critic = self.model
    self.simulator = ActorSimulator(self.Nx, self.Ny, self.actor_critic)
    self.lambda_critic = 0.1
    self.eps = eps

  def get_delta_and_critic_target_old(self, S, R, Sp, term, critic_output_p_old, critic_output_old):
    R_proc = self.R_process(R)
    critic_target_old = R_proc + (1-term) * critic_output_p_old
    delta_old = critic_target_old - critic_output_old
    return delta_old, critic_target_old

  def get_loss_actor(self, actor_output, A, delta_old, actor_output_old):
    r = Game.project(actor_output, A) / Game.project(actor_output_old, A)
    r_clipped = torch.clamp(r, 1-self.eps, 1+self.eps)
    min_arg_1 = r * delta_old
    min_arg_2 = r_clipped * delta_old
    loss_actor = torch.min(min_arg_1, min_arg_2)
    loss_actor = -torch.mean(loss_actor) # minus sign
    return loss_actor


  def get_loss_critic(self, critic_output, critic_target_old):
    loss_critic = nn.MSELoss()(critic_output, critic_target_old)
    return loss_critic


  def train_once(self, N, iterations, batch_size):
    self.simulator.renew_dataset()
    self.simulator.simulate(N = N)

    actor_critic_old = copy.deepcopy(self.actor_critic)
    actor_critic_old.eval() # never put the old model on .train() mode

    dataloader = self.simulator.get_dataloader(batch_size)
    loss_total = 0
    loss_actor_total = 0
    loss_critic_total = 0

    for it in range(iterations):
      for S, A, R, Sp, term, t in dataloader:
        self.optimizer.zero_grad()

        self.actor_critic.train()
        actor_output, critic_output = self.actor_critic(S)

        actor_critic_old.eval()
        actor_output_old, critic_output_old = actor_critic_old(S)
        actor_output_old = actor_output_old.detach()
        critic_output_old = critic_output_old.detach()
        critic_output_p_old = actor_critic_old.critic(Sp).detach()

        delta_old, critic_target_old = self.get_delta_and_critic_target_old(S, R, Sp, term, critic_output_p_old, critic_output_old)

        loss_actor = self.get_loss_actor(actor_output, A, delta_old, actor_output_old)
        loss_critic = self.get_loss_critic(critic_output, critic_target_old)
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

    del actor_critic_old
    gc.collect()
    return loss_mean