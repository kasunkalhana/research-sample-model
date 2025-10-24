"""Wrapper around SUMO's TraCI interface."""
from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

# SUMO libs
from hydra.utils import to_absolute_path  # resolves paths after Hydra changes CWD
from sumolib import checkBinary           # finds sumo/sumo-gui on PATH

try:  # pragma: no cover - optional dependency
    import traci  # type: ignore
except Exception:  # pragma: no cover
    traci = None  # type: ignore


@dataclass
class JunctionState:
    phase: str
    phase_index: int
    time_in_phase: float
    min_green: float
    max_green: float
    queues: Dict[str, float]


@dataclass
class LaneState:
    vehicle_count: int
    mean_speed: float
    density: float
    length: float
    speed_limit: float


@dataclass
class VehicleState:
    lane_id: str
    position: float
    speed: float
    waiting_time: float
    vtype: str


@dataclass
class RawState:
    junctions: Dict[str, JunctionState] = field(default_factory=dict)
    lanes: Dict[str, LaneState] = field(default_factory=dict)
    vehicles: Dict[str, VehicleState] = field(default_factory=dict)
    time: float = 0.0


class TraciClient:
    """Lifecycle manager for SUMO and TraCI subscriptions."""

    def __init__(self, sumocfg: Path, gui: bool = False, step_length: float = 1.0, extra_args: Optional[list[str]] = None):
        self.sumocfg = Path(sumocfg)
        self.gui = gui
        self.step_length = float(step_length)
        self.extra_args = list(extra_args) if extra_args else []
        self._connected: bool = False
        self._subscriptions_initialized = False

    @property
    def connected(self) -> bool:
        return self._connected

    def start(self) -> None:
        """Start SUMO and connect via TraCI."""
        if traci is None:  # pragma: no cover
            raise RuntimeError("TraCI is not available in this environment")

        # Ensure SUMO_HOME is set (nice to have on Windows)
        if "SUMO_HOME" not in os.environ:
            # try to infer from installed traci
            try:
                os.environ["SUMO_HOME"] = str(Path(traci.__file__).resolve().parents[1])
            except Exception:
                pass

        # Resolve config to absolute path because Hydra changes CWD to outputs/...
        cfg_path = to_absolute_path(str(self.sumocfg))

        # Pick binary and assemble command
        binary = checkBinary("sumo-gui" if self.gui else "sumo")
        cmd = [
            binary,
            "-c", cfg_path,
            "--step-length", str(self.step_length),
            "--start",            # start without pressing play
            "--quit-on-end"       # clean exit when done
        ]
        if self.extra_args:
            cmd += self.extra_args

        # Launch SUMO through TraCI (no separate Popen)
        traci.start(cmd, label="gnn")
        self._connected = True
        self._init_subscriptions()

    def stop(self) -> None:
        """Close TraCI session (terminates SUMO it started)."""
        if traci is not None and self._connected:  # pragma: no branch
            with contextlib.suppress(Exception):
                traci.close()
        self._connected = False
        self._subscriptions_initialized = False

    # context manager
    def __enter__(self) -> "TraciClient":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover
        self.stop()

    # ------------------------------------------------------------------ subscriptions
    def _init_subscriptions(self) -> None:
        if traci is None or self._subscriptions_initialized:
            return

        # lanes
        for lane_id in traci.lane.getIDList():
            traci.lane.subscribe(
                lane_id,
                (
                    traci.constants.LAST_STEP_MEAN_SPEED,
                    traci.constants.LAST_STEP_VEHICLE_NUMBER,
                ),
            )

        # traffic lights
        for tls_id in traci.trafficlight.getIDList():
            traci.trafficlight.subscribe(
                tls_id,
                (
                    traci.constants.TL_CURRENT_PHASE,
                    traci.constants.TL_NEXT_SWITCH,
                ),
            )

        # vehicles (subscribe to current ones; new ones will be read ad-hoc)
        for veh_id in traci.vehicle.getIDList():
            traci.vehicle.subscribe(
                veh_id,
                (
                    traci.constants.VAR_LANE_ID,
                    traci.constants.VAR_POSITION,
                    traci.constants.VAR_SPEED,
                    traci.constants.VAR_WAITING_TIME,
                ),
            )

        self._subscriptions_initialized = True

    # ------------------------------------------------------------------ public API
    def step(self) -> RawState:
        """Advance the simulation by one step and return a snapshot."""
        if traci is None:  # pragma: no cover
            raise RuntimeError("TraCI is not available")
        traci.simulationStep()
        sim_time = float(traci.simulation.getTime())
        lanes = self._collect_lanes()
        junctions = self._collect_tls(lanes)
        vehicles = self._collect_vehicles()
        return RawState(junctions=junctions, lanes=lanes, vehicles=vehicles, time=sim_time)

    # ------------------------------ collectors
    def _collect_lanes(self) -> Dict[str, LaneState]:
        if traci is None:
            return {}
        lane_states: Dict[str, LaneState] = {}
        for lane_id in traci.lane.getIDList():
            veh_count = int(traci.lane.getLastStepVehicleNumber(lane_id))
            mean_speed = float(traci.lane.getLastStepMeanSpeed(lane_id))
            length = float(traci.lane.getLength(lane_id))
            speed_limit = float(traci.lane.getMaxSpeed(lane_id))
            density = veh_count / max(length, 1e-3)
            lane_states[lane_id] = LaneState(
                vehicle_count=veh_count,
                mean_speed=mean_speed,
                density=density,
                length=length,
                speed_limit=speed_limit,
            )
        return lane_states

    def _collect_tls(self, lanes: Dict[str, LaneState]) -> Dict[str, JunctionState]:
        if traci is None:
            return {}
        tls_states: Dict[str, JunctionState] = {}
        for tls_id in traci.trafficlight.getIDList():
            phase = traci.trafficlight.getRedYellowGreenState(tls_id)
            phase_index = int(traci.trafficlight.getPhase(tls_id))

            # remaining time and programmed duration
            current_time = float(traci.simulation.getTime())
            next_switch = float(traci.trafficlight.getNextSwitch(tls_id))
            remaining = max(0.0, next_switch - current_time)
            try:
                # duration of the current phase from program
                logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)[0]
                current = logic.phases[phase_index]
                programmed = float(current.duration if current.duration > 0 else current.maxDuration)
            except Exception:
                programmed = float(traci.trafficlight.getPhaseDuration(tls_id))

            time_in_phase = max(0.0, programmed - remaining)  # elapsed time

            # green bounds from program (best-effort)
            try:
                min_green = min(float(p.duration) for p in logic.phases if "y" not in p.state.lower())
                max_green = max(float(p.maxDuration) for p in logic.phases if p.maxDuration > 0)
            except Exception:
                min_green, max_green = 5.0, 60.0

            # queues for controlled lanes
            queues = {
                lane_id: float(lanes[lane_id].vehicle_count)
                for lane_id in traci.trafficlight.getControlledLanes(tls_id)
                if lane_id in lanes
            }

            tls_states[tls_id] = JunctionState(
                phase=phase,
                phase_index=phase_index,
                time_in_phase=time_in_phase,
                min_green=min_green,
                max_green=max_green,
                queues=queues,
            )
        return tls_states

    def _collect_vehicles(self) -> Dict[str, VehicleState]:
        if traci is None:
            return {}
        vehicle_states: Dict[str, VehicleState] = {}
        for veh_id in traci.vehicle.getIDList():
            lane_id = traci.vehicle.getLaneID(veh_id)
            position = float(traci.vehicle.getLanePosition(veh_id))
            speed = float(traci.vehicle.getSpeed(veh_id))
            waiting_time = float(traci.vehicle.getWaitingTime(veh_id))
            vtype = traci.vehicle.getTypeID(veh_id)
            vehicle_states[veh_id] = VehicleState(
                lane_id=lane_id,
                position=position,
                speed=speed,
                waiting_time=waiting_time,
                vtype=vtype,
            )
        return vehicle_states


__all__ = [
    "TraciClient",
    "RawState",
    "JunctionState",
    "LaneState",
    "VehicleState",
]
