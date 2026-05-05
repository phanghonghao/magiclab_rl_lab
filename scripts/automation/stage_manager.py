"""YAML training plan parser and stage transition manager.

A training plan is a YAML file that describes an ordered sequence of stages
(s1 → s5), each with its own config, iteration limit, resume source, and
monitor thresholds.  The :class:`StageManager` reads the plan and provides
the orchestrator with "current stage" / "next stage" queries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class StageConfig:
    """Parsed representation of a single training stage."""

    id: str
    terrain: str
    env_config: str
    max_iterations: int
    num_envs: int = 4096
    resume_from: Optional[str] = None  # stage id to resume from
    initial_checkpoint: Optional[str] = None  # explicit checkpoint path (absolute)
    monitor: Optional[dict] = None     # per-stage monitor overrides
    extra_args: Optional[list[str]] = None


@dataclass
class RetryPolicy:
    """Global retry policy from the training plan."""

    max_retries: int = 2
    nan: Optional[dict] = None
    oom: Optional[dict] = None


class StageManager:
    """Parse a training plan YAML and manage stage transitions."""

    def __init__(self, plan_path: str | Path):
        self._plan_path = Path(plan_path)
        self._stages: list[StageConfig] = []
        self._retry_policy = RetryPolicy()
        self._plan_name = self._plan_path.stem
        self._parse()

    # -- Parsing ------------------------------------------------------------- #

    def _parse(self) -> None:
        if not self._plan_path.exists():
            raise FileNotFoundError(f"Training plan not found: {self._plan_path}")

        with open(self._plan_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

        if not data or "stages" not in data:
            raise ValueError(f"Invalid training plan: missing 'stages' key in {self._plan_path}")

        for raw in data["stages"]:
            stage = StageConfig(
                id=raw["id"],
                terrain=raw.get("terrain", "flat"),
                env_config=raw["env_config"],
                max_iterations=raw.get("max_iterations", 50000),
                num_envs=raw.get("num_envs", 4096),
                resume_from=raw.get("resume_from"),
                initial_checkpoint=raw.get("initial_checkpoint"),
                monitor=raw.get("monitor"),
                extra_args=raw.get("extra_args"),
            )
            self._stages.append(stage)

        # Validate: all resume_from references exist
        stage_ids = {s.id for s in self._stages}
        for s in self._stages:
            if s.resume_from and s.resume_from not in stage_ids:
                raise ValueError(
                    f"Stage '{s.id}' references unknown resume_from='{s.resume_from}'"
                )

        # Parse retry policy
        rp = data.get("retry_policy", {})
        self._retry_policy = RetryPolicy(
            max_retries=rp.get("max_retries", 2),
            nan=rp.get("nan"),
            oom=rp.get("oom"),
        )

        logger.info(
            "Loaded plan '%s' with %d stages: %s",
            self._plan_name,
            len(self._stages),
            [s.id for s in self._stages],
        )

    # -- Queries ------------------------------------------------------------- #

    @property
    def plan_name(self) -> str:
        return self._plan_name

    @property
    def stages(self) -> list[StageConfig]:
        return list(self._stages)

    @property
    def retry_policy(self) -> RetryPolicy:
        return self._retry_policy

    def get_stage_by_id(self, stage_id: str) -> Optional[StageConfig]:
        for s in self._stages:
            if s.id == stage_id:
                return s
        return None

    def get_stage_index(self, stage_id: str) -> int:
        for i, s in enumerate(self._stages):
            if s.id == stage_id:
                return i
        raise ValueError(f"Unknown stage id: {stage_id}")

    def get_current_stage(self, current_stage_id: str) -> Optional[StageConfig]:
        return self.get_stage_by_id(current_stage_id)

    def get_next_stage(self, current_stage_id: str) -> Optional[StageConfig]:
        """Return the stage after *current_stage_id*, or *None* if last."""
        idx = self.get_stage_index(current_stage_id)
        next_idx = idx + 1
        if next_idx < len(self._stages):
            return self._stages[next_idx]
        return None

    def get_resume_checkpoint(
        self,
        stage_id: str,
        stage_history: list[dict],
    ) -> Optional[str]:
        """Resolve the checkpoint path to resume from for *stage_id*.

        Priority:
        1. ``initial_checkpoint`` — explicit path in YAML (for first stage
           resuming from a previous plan's checkpoint)
        2. ``resume_from`` — looks up *stage_history* for a completed entry
        """
        stage = self.get_stage_by_id(stage_id)
        if stage is None:
            return None

        # 1) Explicit checkpoint path
        if stage.initial_checkpoint:
            ckpt = stage.initial_checkpoint
            if Path(ckpt).exists():
                logger.info("Initial checkpoint for '%s': %s", stage_id, ckpt)
                return ckpt
            logger.warning("initial_checkpoint not found: %s", ckpt)

        # 2) resume_from via stage_history
        if stage.resume_from is None:
            return None

        source_id = stage.resume_from
        for entry in reversed(stage_history):
            if entry.get("stage_id") == source_id and entry.get("best_checkpoint_path"):
                ckpt = entry["best_checkpoint_path"]
                if Path(ckpt).exists():
                    logger.info("Resume checkpoint for '%s': %s", stage_id, ckpt)
                    return ckpt
                logger.warning(
                    "Checkpoint for '%s' found in history but file missing: %s",
                    source_id,
                    ckpt,
                )
        logger.warning("No resume checkpoint found for '%s' (source: '%s')", stage_id, source_id)
        return None

    def get_start_stage_id(self, start_from: Optional[str] = None) -> str:
        """Return the stage id to begin execution at.

        If *start_from* is given, validates it exists in the plan.
        Otherwise returns the first stage.
        """
        if start_from:
            if self.get_stage_by_id(start_from) is None:
                raise ValueError(f"--start-from stage '{start_from}' not in plan")
            return start_from
        if not self._stages:
            raise ValueError("Training plan has no stages")
        return self._stages[0].id
