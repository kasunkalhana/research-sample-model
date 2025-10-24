"""Launch the online optimization loop."""

from __future__ import annotations

import time
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from gso.graph.builder import GraphBuilder
from gso.model.hgat import HGATConfig, RecurrentHGAT
from gso.runtime.inference_engine import InferenceConfig, InferenceEngine
from gso.sim.control_translator import ControlConfig, ControlTranslator
from gso.sim.kpis import KPIState
from gso.sim.state_extractor import StateExtractor
from gso.sim.traci_client import TraciClient
from gso.utils.logging import logger, setup_logging

from hydra.utils import to_absolute_path

@hydra.main(config_path="../configs", config_name="inference", version_base=None)
def main(cfg: DictConfig) -> None:
    setup_logging()

    sumocfg = Path(to_absolute_path(cfg.env.sumocfg))
    ckpt_path = Path(to_absolute_path(cfg.model.checkpoint)) if cfg.model.checkpoint else None

    client = TraciClient(sumocfg, gui=cfg.env.gui, step_length=cfg.env.step_length)
    extractor = StateExtractor(cfg.phases)
    builder = GraphBuilder(
        lane_to_junction=dict(cfg.graph.lane_to_junction),
        lane_successors={k: list(v) for k, v in cfg.graph.lane_successors.items()},
    )
    model = RecurrentHGAT(cfg.model.input_dims, HGATConfig(**cfg.model.hgat))

    if ckpt_path and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cpu")
        # try common wrappers
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            logger.warning("Loaded checkpoint with partial match. missing=%s unexpected=%s", missing, unexpected)
    else:
        logger.warning("No valid HGAT checkpoint found at %s; using random init", ckpt_path)

    translator = ControlTranslator(cfg.phases, ControlConfig(**cfg.control))
    engine = InferenceEngine(client, extractor, builder, model, translator, KPIState(), InferenceConfig(**cfg.inference))

    try:
        with client:
            while True:
                result = engine.step()
                logger.info("t={t} actions={actions} kpis={kpis}", t=result["time"], actions=result["actions"], kpis=result["kpis"])
                time.sleep(cfg.loop_sleep)
    except KeyboardInterrupt:
        logger.info("Shutting down inference loop")


if __name__ == "__main__":
    main()
