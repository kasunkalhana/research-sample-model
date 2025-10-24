"""Recurrent heterogeneous graph attention network."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class AttentionAggregator(nn.Module):
    """Simple attention-like aggregator for message passing."""
    def __init__(self, in_src: int, in_dst: int, out: int, dropout: float = 0.1):
        super().__init__()
        self.src_lin = nn.Linear(in_src, out)
        self.dst_lin = nn.Linear(in_dst, out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, dst: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0 or src.numel() == 0 or dst.numel() == 0:
            return torch.zeros((dst.size(0), self.src_lin.out_features), device=dst.device, dtype=dst.dtype)
        src_proj = self.src_lin(src)[edge_index[0]]      # [E, H]
        dst_proj = self.dst_lin(dst)[edge_index[1]]      # [E, H]
        scores = (src_proj * dst_proj).sum(dim=-1)       # [E]
        weights = torch.softmax(scores, dim=0).unsqueeze(-1)  # [E,1]
        weights = self.dropout(weights)
        messages = src_proj * weights                    # [E, H]
        out = torch.zeros((dst.size(0), src_proj.size(-1)), device=dst.device, dtype=dst.dtype)
        out.index_add_(0, edge_index[1], messages)       # aggregate to dst
        return out


@dataclass
class HGATConfig:
    hidden_dim: int = 64
    dropout: float = 0.1
    history: int = 4
    num_actions: int = 4


class RecurrentHGAT(nn.Module):
    """Minimal recurrent HGAT over junction nodes (uses junction/lane features)."""

    def __init__(self, input_dims: Dict[str, int], config: HGATConfig):
        super().__init__()
        self.config = config

        self.encoders = nn.ModuleDict({
            "junction": nn.Sequential(nn.Linear(input_dims["junction"], config.hidden_dim), nn.ReLU()),
            "lane":     nn.Sequential(nn.Linear(input_dims["lane"],     config.hidden_dim), nn.ReLU()),
            "vehicle":  nn.Sequential(nn.Linear(input_dims.get("vehicle", 3),  config.hidden_dim), nn.ReLU()),
        })

        self.aggregators = nn.ModuleDict({
            "lane_to_junction": AttentionAggregator(config.hidden_dim, config.hidden_dim, config.hidden_dim, dropout=config.dropout),
            "junction_self":    AttentionAggregator(config.hidden_dim, config.hidden_dim, config.hidden_dim, dropout=config.dropout),
        })

        self.gru = nn.GRU(config.hidden_dim, config.hidden_dim, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(config.dropout), nn.Linear(config.hidden_dim, config.num_actions))

    def _x_or_zeros(self, data, key: str, feat_dim: int, device: torch.device) -> torch.Tensor:
        if key in data and getattr(data[key], "x", None) is not None:
            return data[key].x
        return torch.zeros((0, feat_dim), device=device, dtype=torch.float32)

    def forward(self, data, hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device

        # 1) node features
        j_in = self.encoders["junction"][0].in_features
        l_in = self.encoders["lane"][0].in_features
        j_x = self._x_or_zeros(data, "junction", j_in, device)   # [N_j, F_j]
        l_x = self._x_or_zeros(data, "lane",     l_in, device)   # [N_l, F_l]

        j_emb = self.encoders["junction"](j_x) if j_x.size(0) > 0 else torch.zeros((0, self.config.hidden_dim), device=device)
        l_emb = self.encoders["lane"](l_x)     if l_x.size(0) > 0 else torch.zeros((0, self.config.hidden_dim), device=device)

        # 2) edges lane->junction
        if ("lane", "to", "junction") in getattr(data, "edge_types", []):
            e_lj = data[("lane", "to", "junction")].edge_index.to(device)
        else:
            e_lj = torch.zeros((2, 0), dtype=torch.long, device=device)

        lane_msgs = self.aggregators["lane_to_junction"](l_emb, j_emb, e_lj) if j_emb.size(0) else torch.zeros_like(j_emb)

        # 3) per-junction combined representation
        combined = j_emb + lane_msgs   # [N_j, H]
        N_j = combined.size(0)
        if N_j == 0:
            # no junction nodes → return a dummy shape for downstream code
            logits = self.head(torch.zeros((1, self.config.hidden_dim), device=device))
            return logits, None

        # 4) recurrent head per junction (one step)
        # GRU(batch_first=True): input [B, T, H], hidden [num_layers, B, H]
        seq = combined.unsqueeze(1)  # [N_j, 1, H]
        if hidden is None or hidden.size(1) != N_j:
            hidden = torch.zeros((1, N_j, self.config.hidden_dim), device=device)

        out, new_hidden = self.gru(seq, hidden)  # out: [N_j, 1, H]
        z = out[:, -1, :]                        # [N_j, H]
        logits = self.head(z)                    # [N_j, A]
        return logits, new_hidden
