"""Entry point for MARL PPO fine-tuning."""

from __future__ import annotations

from pathlib import Path
import inspect

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

from gso.training.marl_env import JunctionMARLEnv
from gso.training.marl_trainer import PPOConfig, PPOTrainer
from gso.utils.logging import setup_logging


@hydra.main(config_path="../configs", config_name="train_marl", version_base=None)
def main(cfg: DictConfig) -> None:
    setup_logging()

    # Normalize any file paths you might use later:
    sumocfg_path = Path(to_absolute_path(cfg.env.sumocfg_path))

    # Minimal toy env (replace with real SUMO env when integrating)
    env = JunctionMARLEnv(num_junctions=cfg.env.num_junctions, action_dim=cfg.model.action_dim)
    actor = torch.nn.Linear(cfg.env.state_dim, cfg.model.action_dim)
    critic = torch.nn.Linear(cfg.env.state_dim, 1)

    # Convert trainer section to a plain dict
    trainer_cfg = OmegaConf.to_container(cfg.trainer, resolve=True)  # type: ignore[assignment]
    assert isinstance(trainer_cfg, dict)

    # Pull out fields not meant for PPOConfig
    episodes = int(trainer_cfg.pop("episodes", 100))  # used only by trainer.train()

    # Auto-filter unknown kwargs so PPOConfig(**kwargs) won't crash
    ppo_sig = inspect.signature(PPOConfig)
    allowed = set(ppo_sig.parameters.keys())
    ppo_kwargs = {k: v for k, v in trainer_cfg.items() if k in allowed}

    # Build config & trainer
    ppo_cfg = PPOConfig(**ppo_kwargs)
    trainer = PPOTrainer(env, actor, critic, ppo_cfg)

    # Train
    trainer.train(episodes=episodes)

    # Save checkpoints (Hydra-safe path)
    ckpt_path = Path(to_absolute_path(cfg.checkpoint))
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict()}, ckpt_path)
    torch.save(actor.state_dict(), "checkpoints/hgat_policy.pt")


if __name__ == "__main__":
    main()
