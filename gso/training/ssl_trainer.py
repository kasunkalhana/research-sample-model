"""Self-supervised training loop."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.logging import logger
from .ssl_dataset import SSLDataset, ssl_collate


@dataclass
class SSLTrainConfig:
    epochs: int = 5
    lr: float = 1e-3
    batch_size: int = 32
    history: int = 4


class SSLTrainer:
    """Train a model to predict next-step junction features."""

    def __init__(self, model: torch.nn.Module, config: SSLTrainConfig):
        self.model = model
        self.config = config
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    def fit(self, dataset: SSLDataset) -> None:
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=ssl_collate,  # << important
        )
        self.model.train()
        for epoch in range(self.config.epochs):
            losses = []
            for H, T in tqdm(loader, desc=f"ssl-epoch-{epoch}"):
                # H: [B, K*F], T: [B, F]
                pred = self.model(H)          # [B, F]
                loss = F.mse_loss(pred, T)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.append(float(loss.detach()))
            avg = sum(losses) / max(len(losses), 1)
            logger.info("SSL epoch {epoch} loss={loss:.6f}", epoch=epoch, loss=avg)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)


__all__ = ["SSLTrainer", "SSLTrainConfig"]
