"""Entry point for self-supervised pretraining."""

from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from gso.runtime.recorder import RecorderConfig, RolloutRecorder
from gso.sim.state_extractor import StateExtractor
from gso.sim.traci_client import TraciClient
from gso.training.ssl_dataset import SSLExample, load_ssl_dataset
from gso.training.ssl_trainer import SSLTrainer, SSLTrainConfig
from gso.utils.logging import setup_logging


@hydra.main(config_path="../configs", config_name="train_ssl", version_base=None)
def main(cfg: DictConfig) -> None:
    setup_logging()
    mode = cfg.get("mode", "train")

    if mode == "record":
        # Resolve paths because Hydra changes CWD to outputs/...
        sumocfg = Path(to_absolute_path(cfg.env.sumocfg))
        out_path = Path(to_absolute_path(cfg.output))
        extractor = StateExtractor(cfg.phases)
        client = TraciClient(sumocfg, gui=bool(cfg.env.gui), step_length=float(cfg.env.step_length))
        recorder = RolloutRecorder(client, extractor, RecorderConfig(steps=int(cfg.steps), output=out_path))
        recorder.run()
        return

    # TRAIN mode -----------------------------------------------------------
    ds_path = Path(to_absolute_path(cfg.dataset))
    dataset = load_ssl_dataset(ds_path, history=int(cfg.trainer.history))

    # Determine model I/O dims from a sample
    ex0: SSLExample = dataset[0]
    in_dim = int(np.asarray(ex0.history).size)   # K*F
    out_dim = int(np.asarray(ex0.target).size)   # F

    model = torch.nn.Sequential(
        torch.nn.Linear(in_dim, int(cfg.model.hidden_dim)),
        torch.nn.ReLU(),
        torch.nn.Linear(int(cfg.model.hidden_dim), out_dim),
    )

    trainer = SSLTrainer(model, SSLTrainConfig(**cfg.trainer))
    trainer.fit(dataset)

    ckpt_path = Path(to_absolute_path(cfg.checkpoint))
    trainer.save(ckpt_path)


if __name__ == "__main__":
    main()
