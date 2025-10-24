"""PPO-style trainer for MARL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from ..utils.logging import logger
from .marl_env import JunctionMARLEnv


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    epochs: int = 3
    batch_size: int = 32  # (kept for API symmetry; we use full-trajectory updates here)
    max_steps: int = 200
    value_coef: float = 0.5
    entropy_coef: float = 0.01


class PPOTrainer:
    def __init__(self, env: JunctionMARLEnv, actor: torch.nn.Module, critic: torch.nn.Module, config: PPOConfig):
        self.env = env
        self.actor = actor
        self.critic = critic
        self.config = config
        self.actor_optim = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)
        self.critic_optim = torch.optim.Adam(critic.parameters(), lr=config.critic_lr)

    @torch.no_grad()
    def run_episode(self) -> Dict[str, torch.Tensor]:
        """Collect one on-policy trajectory under the CURRENT policy.
        Everything returned here is DETACHED (no grad graphs)."""
        obs = self.env.reset()

        observations: List[torch.Tensor] = []
        actions: List[torch.Tensor] = []
        old_log_probs: List[torch.Tensor] = []
        rewards: List[torch.Tensor] = []
        values: List[torch.Tensor] = []

        for _ in range(self.config.max_steps):
            logits = self.actor(obs)                           # [A]
            dist = Categorical(logits=logits)
            action = dist.sample()                             # []
            logp = dist.log_prob(action)                       # []
            value = self.critic(obs).squeeze(-1)               # []

            next_obs, reward, done, _ = self.env.step(action.float())

            observations.append(obs.detach())
            actions.append(action.detach())
            old_log_probs.append(logp.detach())
            rewards.append(reward.detach() if torch.is_tensor(reward) else torch.tensor(reward, dtype=torch.float32))
            values.append(value.detach())

            obs = next_obs
            if done:
                break

        # Stack into [T, ...]
        return {
            "observations": torch.stack(observations),   # [T, state_dim]
            "actions": torch.stack(actions),             # [T]
            "old_log_probs": torch.stack(old_log_probs), # [T]
            "rewards": torch.stack(rewards),             # [T]
            "values": torch.stack(values),               # [T]
        }

    @staticmethod
    def _compute_advantages(rewards: torch.Tensor, values: torch.Tensor, gamma: float, lam: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """GAE(λ), keeping everything as tensors (no Python floats)."""
        # Ensure 1D float tensors
        rewards = rewards.detach().to(dtype=torch.float32)
        values = values.detach().to(dtype=torch.float32)

        T = rewards.size(0)
        adv = torch.zeros_like(rewards)
        last_gae = torch.zeros((), dtype=rewards.dtype, device=rewards.device)  # scalar tensor 0.0

        for t in reversed(range(T)):
            v_t = values[t]
            v_tp1 = values[t + 1] if t + 1 < T else torch.zeros((), dtype=values.dtype, device=values.device)
            delta = rewards[t] + gamma * v_tp1 - v_t                      # tensor scalar
            last_gae = delta + gamma * lam * last_gae                     # tensor scalar
            adv[t] = last_gae

        returns = adv + values
        return adv, returns

    def train(self, episodes: int = 10) -> None:
        for episode in range(episodes):
            batch = self.run_episode()

            # Advantages/returns computed from DETACHED buffers
            advantages, returns = self._compute_advantages(
                batch["rewards"], batch["values"], self.config.gamma, self.config.gae_lambda
            )
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Multiple epochs over the same on-policy batch
            for _ in range(self.config.epochs):
                # Recompute new log-probs and values from CURRENT policy over the WHOLE trajectory
                logits = self.actor(batch["observations"])                 # [T, A]
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(batch["actions"])            # [T]
                entropy = dist.entropy().mean()                            # scalar

                ratio = (new_log_probs - batch["old_log_probs"]).exp()     # [T]
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio) * advantages
                policy_loss = -(torch.min(surr1, surr2)).mean()            # scalar

                value_pred = self.critic(batch["observations"]).squeeze(-1)  # [T]
                value_loss = F.mse_loss(value_pred, returns)                 # scalar (mean)

                # Update actor
                self.actor_optim.zero_grad()
                (policy_loss - self.config.entropy_coef * entropy).backward()
                self.actor_optim.step()

                # Update critic
                self.critic_optim.zero_grad()
                (self.config.value_coef * value_loss).backward()
                self.critic_optim.step()

            logger.info(
                "ppo episode {episode} loss_pi={pi:.6f} loss_v={vl:.6f}",
                episode=episode,
                pi=float(policy_loss.detach()),
                vl=float(value_loss.detach()),
            )
