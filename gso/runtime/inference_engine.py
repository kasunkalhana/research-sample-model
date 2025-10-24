"""Runtime inference loop tying together simulation and policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from ..graph.builder import GraphBuilder
from ..model.hgat import RecurrentHGAT
from ..model.mc_dropout import mc_dropout_inference, predictive_entropy
from ..model.policy import select_actions
from ..sim.control_translator import ControlTranslator
from ..sim.kpis import KPIState
from ..sim.state_extractor import StateExtractor
from ..sim.traci_client import RawState, TraciClient


@dataclass
class InferenceConfig:
    mc_passes: int = 10


class InferenceEngine:
    def __init__(
        self,
        client: TraciClient,
        extractor: StateExtractor,
        builder: GraphBuilder,
        model: RecurrentHGAT,
        translator: ControlTranslator,
        kpis: KPIState,
        config: InferenceConfig,
    ):
        self.client = client
        self.extractor = extractor
        self.builder = builder
        self.model = model
        self.translator = translator
        self.kpis = kpis
        self.config = config
        self.hidden: Optional[torch.Tensor] = None

    def _to_device(self, data):
        """Move node features (and optionally edge_index) onto the model's device."""
        device = next(self.model.parameters()).device
        # node features
        for nt in getattr(data, "node_types", []):
            if getattr(data[nt], "x", None) is not None:
                data[nt].x = data[nt].x.to(device)
            if hasattr(data[nt], "mask") and data[nt].mask is not None:
                data[nt].mask = data[nt].mask.to(device)
        # edges (PyG keeps edge_index on CPU fine, but moving is ok)
        for et in getattr(data, "edge_types", []):
            ei = getattr(data[et], "edge_index", None)
            if ei is not None:
                data[et].edge_index = ei.to(device)
        return data

    def step(self) -> Dict[str, object]:
        raw = self.client.step()
        struct = self.extractor.extract(raw)
        data = self.builder.build(struct, raw.time)
        data = self._to_device(data)

        N_j = data["junction"].x.size(0) if hasattr(data["junction"], "x") and data["junction"].x is not None else 0
        if self.hidden is not None and self.hidden.size(1) != max(N_j, 1):
            self.hidden = None

        def forward_fn(model: RecurrentHGAT) -> torch.Tensor:
            logits, _ = model(data, self.hidden)
            return logits

        mean_logits, var_logits = mc_dropout_inference(self.model, forward_fn, passes=self.config.mc_passes)

        # Deterministic pass for action selection (also updates RNN hidden state)
        self.model.eval()
        logits, self.hidden = self.model(data, self.hidden)   # shape [N_j, A]

        # --- Align node_order with available logits rows
        jids = self.builder.node_order.get("junction", [])
        N = min(len(jids), logits.size(0))
        jidsN = jids[:N]

        # Uncertainty for the N rows we have
        uncertainties = predictive_entropy(mean_logits)        # shape [N_j]
        uncertainties = uncertainties[:N].detach().cpu()

        # Build legal masks for exactly these jids
        legal_masks = {
            jid: torch.tensor(struct.legal_actions.get(jid, [1] * logits.size(1)),
                              dtype=torch.float32,
                              device=logits.device)
            for jid in jidsN
        }

        # Select actions only for the aligned set
        actions = select_actions(logits[:N], legal_masks, {"junction": jidsN})

        # Translate with aligned uncertainties
        action_dict = self.translator.translate(
            actions,
            {jid: float(uncertainties[i]) for i, jid in enumerate(jidsN)},
            raw.junctions,
        )

        self.kpis.update(raw)
        return {
            "time": raw.time,
            "actions": action_dict,
            "uncertainty": {jid: float(uncertainties[i]) for i, jid in enumerate(jidsN)},
            "kpis": self.kpis.as_dict(),
        }
