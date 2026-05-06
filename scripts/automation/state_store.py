"""Crash-recovery state store using atomic JSON writes.

The orchestrator persists its current state to ``orchestrator_state.json``
after every meaningful transition.  On startup the file is read to decide
whether to resume an in-progress stage or start fresh.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class OrchestratorState:
    """Serializable snapshot of the orchestrator's progress."""

    plan_name: str = ""
    current_stage_id: str = ""          # e.g. "s4_rough_l1" (now = sub_phase id)
    current_stage_status: str = "pending"  # pending / running / overfitting / complete / failed
    training_pid: Optional[int] = None
    training_run_dir: Optional[str] = None
    best_checkpoint_path: Optional[str] = None
    best_reward: Optional[float] = None
    stage_history: list[dict] = field(default_factory=list)  # completed stage results
    retry_count: int = 0
    started_at: str = ""
    updated_at: str = ""

    # Phase-based fields (new)
    current_phase_id: str = ""          # e.g. "p1"
    starting_reward: Optional[float] = None  # reward at start of sub-phase (for rollback)
    rollback_count: int = 0             # per sub-phase rollback counter
    phase_history: list[dict] = field(default_factory=list)  # completed phase results

    def touch(self) -> None:
        """Refresh ``updated_at`` timestamp."""
        self.updated_at = datetime.now().isoformat()


class StateStore:
    """Atomic-read / atomic-write JSON persistence for :class:`OrchestratorState`."""

    def __init__(self, path: str | Path = "orchestrator_state.json"):
        self._path = Path(path)

    # -- Read ---------------------------------------------------------------- #

    def load(self) -> Optional[OrchestratorState]:
        """Load state from disk.  Returns *None* if file does not exist or is
        corrupt."""
        if not self._path.exists():
            return None
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            # Filter out unknown keys gracefully
            valid_keys = {f.name for f in OrchestratorState.__dataclass_fields__.values()}
            filtered = {k: v for k, v in data.items() if k in valid_keys}
            return OrchestratorState(**filtered)
        except Exception:
            return None

    # -- Write --------------------------------------------------------------- #

    def save(self, state: OrchestratorState) -> None:
        """Atomically write *state* to disk (write-to-tmp + rename)."""
        state.touch()
        payload = asdict(state)
        blob = json.dumps(payload, indent=2, ensure_ascii=False)

        # Atomic write: tmp file in same directory, then os.replace
        dir_path = self._path.parent
        dir_path.mkdir(parents=True, exist_ok=True)

        fd, tmp_path = tempfile.mkstemp(dir=str(dir_path), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(blob)
            os.replace(tmp_path, str(self._path))
        except BaseException:
            # Clean up partial tmp file on any error
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    # -- Delete -------------------------------------------------------------- #

    def clear(self) -> None:
        """Remove the state file (used with ``--fresh``)."""
        try:
            self._path.unlink()
        except FileNotFoundError:
            pass
