"""Phase-based training plan parser with three-layer config merging.

Reads the new YAML format (phases → sub_phases) and provides:
  - ``PhaseConfig`` / ``SubPhaseConfig`` dataclasses
  - Three-layer merge: ``defaults`` → ``phase`` → ``sub_phase``
  - Ordered iteration over all 10 sub-phases
  - Checkpoint resolution from phase history
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class SubPhaseConfig:
    """Fully-resolved configuration for a single sub-phase (e.g. p1_coarse)."""

    id: str
    name: str
    phase_id: str
    terrain: str  # flat / gentle / rough — for monitor presets

    # Training params
    max_iterations: int
    num_envs: int

    # Fully merged config dicts
    env: dict = field(default_factory=dict)
    ppo: dict = field(default_factory=dict)
    rewards: dict = field(default_factory=dict)
    monitor: dict = field(default_factory=dict)
    rollback: dict = field(default_factory=dict)


@dataclass
class PhaseConfig:
    """A single phase containing an ordered list of sub-phases."""

    id: str
    name: str
    terrain: str
    sub_phases: list[SubPhaseConfig] = field(default_factory=list)


class PhaseManager:
    """Parse a 5-phase YAML training plan and manage config merging."""

    def __init__(self, plan_path: str | Path):
        self._plan_path = Path(plan_path)
        self._plan_name: str = ""
        self._defaults: dict = {}
        self._phases: list[PhaseConfig] = []
        self._all_sub_phases: list[SubPhaseConfig] = []
        self._num_gpus: int = 4
        self._num_envs: int = 4096
        self._task: str = "Magiclab-Z1-12dof-Velocity"
        self._parse()

    # ── Parsing ─────────────────────────────────────────────────── #

    def _parse(self) -> None:
        if not self._plan_path.exists():
            raise FileNotFoundError(f"Training plan not found: {self._plan_path}")

        with open(self._plan_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

        if not data or "phases" not in data:
            raise ValueError(f"Invalid training plan: missing 'phases' key in {self._plan_path}")

        self._plan_name = data.get("plan_name", self._plan_path.stem)
        self._defaults = data.get("defaults", {})
        self._num_gpus = data.get("num_gpus", 4)
        self._num_envs = data.get("num_envs", 4096)
        self._task = data.get("task", "Magiclab-Z1-12dof-Velocity")

        for raw_phase in data["phases"]:
            phase = self._parse_phase(raw_phase)
            self._phases.append(phase)
            self._all_sub_phases.extend(phase.sub_phases)

        logger.info(
            "Loaded plan '%s': %d phases, %d sub-phases: %s",
            self._plan_name,
            len(self._phases),
            len(self._all_sub_phases),
            [sp.id for sp in self._all_sub_phases],
        )

    def _parse_phase(self, raw: dict) -> PhaseConfig:
        phase_id = raw["id"]
        phase_name = raw.get("name", phase_id)
        terrain = raw.get("terrain", "flat")

        # Phase-level env/ppo/rewards/monitor (partial overrides)
        phase_env = raw.get("env", {})
        phase_ppo = raw.get("ppo", {})
        phase_rewards = raw.get("rewards", {})
        phase_monitor = raw.get("monitor", {})
        phase_rollback = raw.get("rollback", {})

        phase = PhaseConfig(id=phase_id, name=phase_name, terrain=terrain)

        for raw_sp in raw.get("sub_phases", []):
            sp = self._parse_sub_phase(
                raw_sp, phase_id, terrain,
                phase_env, phase_ppo, phase_rewards, phase_monitor, phase_rollback,
            )
            phase.sub_phases.append(sp)

        return phase

    def _parse_sub_phase(
        self,
        raw: dict,
        phase_id: str,
        terrain: str,
        phase_env: dict,
        phase_ppo: dict,
        phase_rewards: dict,
        phase_monitor: dict,
        phase_rollback: dict,
    ) -> SubPhaseConfig:
        sp_id = raw["id"]
        sp_name = raw.get("name", sp_id)

        # Three-layer merge: defaults → phase → sub_phase
        merged_env = _deep_merge(
            _deep_merge(self._defaults.get("env", {}), phase_env),
            raw.get("env", {}),
        )
        merged_ppo = _deep_merge(
            _deep_merge(self._defaults.get("ppo", {}), phase_ppo),
            raw.get("ppo", {}),
        )
        merged_rewards = _deep_merge(
            _deep_merge(self._defaults.get("rewards", {}), phase_rewards),
            raw.get("rewards", {}),
        )
        merged_monitor = _deep_merge(
            _deep_merge(self._defaults.get("monitor", {}), phase_monitor),
            raw.get("monitor", {}),
        )
        merged_rollback = _deep_merge(
            _deep_merge(self._defaults.get("rollback", {}), phase_rollback),
            raw.get("rollback", {}),
        )

        return SubPhaseConfig(
            id=sp_id,
            name=sp_name,
            phase_id=phase_id,
            terrain=terrain,
            max_iterations=raw.get("max_iterations", 10000),
            num_envs=raw.get("num_envs", self._num_envs),
            env=merged_env,
            ppo=merged_ppo,
            rewards=merged_rewards,
            monitor=merged_monitor,
            rollback=merged_rollback,
        )

    # ── Properties ──────────────────────────────────────────────── #

    @property
    def plan_name(self) -> str:
        return self._plan_name

    @property
    def num_gpus(self) -> int:
        return self._num_gpus

    @property
    def task(self) -> str:
        return self._task

    @property
    def phases(self) -> list[PhaseConfig]:
        return list(self._phases)

    @property
    def all_sub_phases(self) -> list[SubPhaseConfig]:
        return list(self._all_sub_phases)

    @property
    def defaults(self) -> dict:
        return copy.deepcopy(self._defaults)

    # ── Queries ─────────────────────────────────────────────────── #

    def get_sub_phase(self, sp_id: str) -> Optional[SubPhaseConfig]:
        for sp in self._all_sub_phases:
            if sp.id == sp_id:
                return sp
        return None

    def get_sub_phase_index(self, sp_id: str) -> int:
        for i, sp in enumerate(self._all_sub_phases):
            if sp.id == sp_id:
                return i
        raise ValueError(f"Unknown sub-phase id: {sp_id}")

    def get_next_sub_phase(self, current_id: str) -> Optional[SubPhaseConfig]:
        idx = self.get_sub_phase_index(current_id)
        next_idx = idx + 1
        if next_idx < len(self._all_sub_phases):
            return self._all_sub_phases[next_idx]
        return None

    def get_start_sub_phase_id(self, start_from: Optional[str] = None) -> str:
        if start_from:
            if self.get_sub_phase(start_from) is None:
                raise ValueError(f"--start-from sub-phase '{start_from}' not in plan")
            return start_from
        if not self._all_sub_phases:
            raise ValueError("Training plan has no sub-phases")
        return self._all_sub_phases[0].id

    def get_phase_for_sub_phase(self, sp_id: str) -> Optional[PhaseConfig]:
        for phase in self._phases:
            for sp in phase.sub_phases:
                if sp.id == sp_id:
                    return phase
        return None


# ── Helpers ──────────────────────────────────────────────────────── #


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (override wins on conflicts)."""
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result
