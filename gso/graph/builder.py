"""Build PyTorch Geometric :class:`HeteroData` objects."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import torch

from ..sim.state_extractor import StructuredState
from .schema import GraphSchema, DEFAULT_SCHEMA

try:
    from torch_geometric.data import HeteroData
except Exception:
    # Very small stand-in for tests if PyG is absent
    class _NodeStore(dict):
        def __init__(self): super().__init__(); self.x=None; self.mask=None
    class _EdgeStore(dict): pass
    class HeteroData(dict):  # type: ignore
        def __init__(self):
            super().__init__()
            self._edge = {}
            self.node_types = []
            self.edge_types = []
        def __getitem__(self, key):
            if isinstance(key, tuple):  # edge type
                if key not in self._edge: self._edge[key] = _EdgeStore()
                return self._edge[key]
            else:
                if key not in self: self[key] = _NodeStore(); self.node_types.append(key)
                return dict.__getitem__(self, key)

class GraphBuilder:
    """Incrementally build :class:`HeteroData` objects from structured state."""

    def __init__(
        self,
        schema: GraphSchema = DEFAULT_SCHEMA,
        lane_to_junction: Dict[str, str] | None = None,
        lane_successors: Dict[str, Iterable[str]] | None = None,
    ):
        self.schema = schema
        self.lane_to_junction = lane_to_junction or {}
        self.lane_successors = lane_successors or {}
        self.node_to_index: Dict[str, Dict[str, int]] = defaultdict(dict)
        self.node_order: Dict[str, List[str]] = defaultdict(list)

    def _ensure_nodes(self, node_type: str, ids: Iterable[str]) -> None:
        mapping = self.node_to_index[node_type]
        order = self.node_order[node_type]
        for node_id in ids:
            if node_id not in mapping:
                mapping[node_id] = len(order)
                order.append(node_id)

    def _make_edges(self, pairs: List[Tuple[int, int]]) -> torch.Tensor:
        if not pairs:
            return torch.zeros((2, 0), dtype=torch.long)
        return torch.tensor(pairs, dtype=torch.long).t().contiguous()

    def build(self, state: StructuredState, time: float = 0.0) -> HeteroData:
        """Create a :class:`HeteroData` snapshot."""
        self._ensure_nodes("junction", state.junction_features.keys())
        self._ensure_nodes("lane",     state.lane_features.keys())
        self._ensure_nodes("vehicle",  state.vehicle_features.keys())

        data = HeteroData()
        feature_dims = self.schema.feature_dims()

        # ---- Node features & masks
        for node_type in ["junction", "lane", "vehicle"]:
            # infer dim if schema says 0
            dim_schema = feature_dims.get(node_type, 0)
            feat_dict = getattr(state, f"{node_type}_features")
            inferred = 0
            for _, arr in feat_dict.items():
                inferred = max(inferred, int(getattr(arr, "shape", [len(arr)])[0] if hasattr(arr, "shape") else len(arr)))
            dim = dim_schema if dim_schema > 0 else inferred

            order = self.node_order[node_type]
            feats = torch.zeros((len(order), max(1, dim)), dtype=torch.float32)  # avoid 0 columns
            mask = torch.zeros(len(order), dtype=torch.bool)

            for node_id, arr in feat_dict.items():
                idx = self.node_to_index[node_type][node_id]
                arr_t = torch.as_tensor(arr, dtype=torch.float32).view(-1)
                L = min(arr_t.numel(), feats.size(1))
                feats[idx, : L] = arr_t[:L]
                mask[idx] = True

            data[node_type].x = feats
            data[node_type].mask = mask

        # ---- Edges
        lane_idx    = self.node_to_index["lane"]
        junction_idx= self.node_to_index["junction"]

        # lane -> junction edges (by mapping)
        pairs_lj = [
            (lane_idx[l], junction_idx[self.lane_to_junction[l]])
            for l in state.lane_features
            if l in self.lane_to_junction and self.lane_to_junction[l] in junction_idx
        ]
        data[("lane", "to", "junction")].edge_index = self._make_edges(pairs_lj)

        # junction -> lane edges (reverse mapping)
        pairs_jl = [
            (junction_idx[j], lane_idx[l])
            for l, j in self.lane_to_junction.items()
            if l in lane_idx and j in junction_idx
        ]
        data[("junction", "to", "lane")].edge_index = self._make_edges(pairs_jl)

        # lane -> lane successors
        pairs_ll = [
            (lane_idx[s], lane_idx[d])
            for s, dsts in self.lane_successors.items() if s in lane_idx
            for d in dsts if d in lane_idx
        ]
        data[("lane", "to", "lane")].edge_index = self._make_edges(pairs_ll)

        # (optional) vehicle -> lane if you ever use it
        data[("vehicle", "to", "lane")].edge_index = self._make_edges([])

        # misc
        data["time"] = torch.tensor([time], dtype=torch.float32)
        return data
