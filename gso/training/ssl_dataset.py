"""Self-supervised dataset utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class SSLExample:
    """One SSL sample: K-step history -> next-step target."""
    history: torch.Tensor  # shape: [K, F] (float32)
    target: torch.Tensor   # shape: [F]     (float32)


class SSLDataset(Dataset):
    """Construct sequences from recorded junction features."""

    def __init__(self, dataframe: pd.DataFrame, history: int = 4):
        self.history = int(history)
        grouped = dataframe.groupby("junction_id")
        samples: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for _, group in grouped:
            # Each row has a vector 'features' (list/np.array). Stack into [T, F]
            feats = np.stack(group["features"].to_list()).astype(np.float32)
            feats_t = torch.from_numpy(feats)  # [T, F]
            T = feats_t.size(0)
            for idx in range(T - self.history):
                hist = feats_t[idx : idx + self.history]   # [K, F]
                tgt = feats_t[idx + self.history]          # [F]
                samples.append((hist, tgt))
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> SSLExample:
        hist, target = self.samples[idx]
        return SSLExample(history=hist, target=target)


def ssl_collate(batch: List[SSLExample]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Turn a list[SSLExample] into batched tensors.
    Returns:
        H: [B, K*F]  (flattened history)
        T: [B, F]
    """
    H_list, T_list = [], []
    for ex in batch:
        h = ex.history.reshape(-1)    # [K*F]
        t = ex.target.reshape(-1)     # [F]
        H_list.append(h)
        T_list.append(t)
    H = torch.stack(H_list, dim=0).contiguous()
    T = torch.stack(T_list, dim=0).contiguous()
    return H, T


def load_ssl_dataset(path: Path, history: int = 4) -> "SSLDataset":
    df = pd.read_parquet(path)
    return SSLDataset(df, history=history)


__all__ = ["SSLDataset", "SSLExample", "ssl_collate", "load_ssl_dataset"]
