"""Embedded training monitor that reuses ``train_monitor.py`` detection logic.

This wraps the existing :class:`OverfittingDetector`, :class:`BestModelTracker`,
:class:`TensorBoardParser`, and :class:`CheckpointAnalyzer` from
``train_monitor.py`` and adds a polling interface suitable for the
orchestrator's event loop.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

# Re-export from train_monitor (sibling package)
import sys

_scripts_dir = str(Path(__file__).resolve().parent.parent)
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from train_monitor import (  # noqa: E402
    BestModelTracker,
    CheckpointAnalyzer,
    MonitorConfig,
    OverfittingDetector,
    RunState,
    TensorBoardParser,
    TAG_ACTION_RATE,
    TAG_ENTROPY,
    TAG_REWARD,
    TAG_VALUE_LOSS,
)

logger = logging.getLogger(__name__)


class EmbeddedMonitor:
    """Incrementally monitor a single training run and invoke a callback when
    overfitting is detected.

    Parameters
    ----------
    log_root:
        Root log directory (e.g. ``logs/rsl_rl/magiclab_z1_12dof_velocity``).
    terrain_type:
        One of ``"flat"``, ``"gentle"``, ``"rough"`` — selects threshold presets.
    on_overfitting:
        Callback invoked **once** when overfitting is first detected.
        Receives the :class:`RunState` as its sole argument.
    action_rate_threshold:
        Override the terrain-preset action-rate threshold.
    min_iterations:
        Minimum iterations before overfitting checks begin.
    """

    def __init__(
        self,
        log_root: str,
        terrain_type: str = "gentle",
        on_overfitting: Optional[Callable[[RunState], None]] = None,
        action_rate_threshold: float = -1.0,
        min_iterations: int = 2000,
        reward_decline_pct: float = 20.0,
        save_interval: int = 100,
    ):
        self._cfg = MonitorConfig(
            log_root=log_root,
            terrain_type=terrain_type,
            action_rate_threshold=action_rate_threshold,
            min_iterations=min_iterations,
            reward_decline_pct=reward_decline_pct,
        )
        self._detector = OverfittingDetector(self._cfg)
        self._tracker = BestModelTracker(save_interval=save_interval)
        self._on_overfitting = on_overfitting

        self._state: Optional[RunState] = None
        self._overfitting_fired = False
        self._phase_start_iter: Optional[int] = None

    # -- Setup --------------------------------------------------------------- #

    def start(self, run_dir: str) -> None:
        """Initialise monitoring for *run_dir* (called once when training starts).

        The *min_iterations* threshold is interpreted as **relative to the first
        observed iteration** — so if the run resumes from iter 5000 and
        min_iterations is 2000, overfitting checks begin at iter 7000.
        """
        run_name = Path(run_dir).name
        self._state = RunState(run_name=run_name, run_dir=run_dir)
        self._overfitting_fired = False
        self._phase_start_iter = None  # set on first data in poll()
        self._baseline_iter: Optional[int] = None  # first observed iter
        logger.info("Monitor started for run: %s", run_name)

    # -- Phase reset --------------------------------------------------------- #

    def reset_for_new_phase(self) -> None:
        """Reset monitor state for a new sub-phase.

        Called by the orchestrator when switching to a new sub-phase so that
        iteration counters (``_phase_start_iter``) and overfitting flags are
        cleared, even though the underlying ``RunState`` may already carry
        residual data from a resumed run.
        """
        self._phase_start_iter = None
        self._overfitting_fired = False
        if self._state is not None:
            self._state.overfitting_detected = False
            self._state.overfitting_reason = None
        logger.info("Monitor reset for new sub-phase")

    # -- Poll ---------------------------------------------------------------- #

    def poll(self) -> dict:
        """Read new TensorBoard data + checkpoints and run overfitting detection.

        Returns a summary dict::

            {
                "status": "HEALTHY" | "OVERFITTING" | "NO_DATA",
                "latest_iter": int,
                "latest_reward": float,
                "peak_reward": float,
                "best_model_iter": int,
                "best_model_reward": float,
                "best_checkpoint_path": str | None,
                "reason": str | None,
            }
        """
        if self._state is None:
            return {"status": "NO_DATA"}

        state = self._state

        # Incremental TensorBoard read
        event_dir = TensorBoardParser.find_event_file(state.run_dir)
        if event_dir:
            try:
                metrics = TensorBoardParser.read_metrics(
                    event_dir, last_step=state.last_checked_step
                )
                state.rewards.extend(metrics.get(TAG_REWARD, []))
                state.action_rates.extend(metrics.get(TAG_ACTION_RATE, []))
                state.value_losses.extend(metrics.get(TAG_VALUE_LOSS, []))
                state.entropies.extend(metrics.get(TAG_ENTROPY, []))
            except Exception as exc:
                logger.warning("TB parse error for %s: %s", state.run_name, exc)

        if state.rewards:
            state.last_checked_step = state.rewards[-1][0]

        # Incremental checkpoint scan
        all_ckpts = CheckpointAnalyzer.find_checkpoints(state.run_dir)
        state.checkpoints = all_ckpts
        for ckpt_path in all_ckpts:
            iteration = CheckpointAnalyzer.get_iteration(ckpt_path)
            if iteration > state.last_checked_ckpt_iter and iteration not in state.std_values:
                try:
                    state.std_values[iteration] = CheckpointAnalyzer.extract_std(ckpt_path)
                except Exception:
                    pass
        if all_ckpts:
            state.last_checked_ckpt_iter = max(
                CheckpointAnalyzer.get_iteration(c) for c in all_ckpts
            )

        # Update peak / best tracking
        for step, reward in state.rewards:
            if reward > state.peak_reward:
                state.peak_reward = reward
                state.peak_reward_iter = step
        for _, ent in state.entropies:
            if ent > state.peak_entropy:
                state.peak_entropy = ent
        self._tracker.update(state)

        # Record baseline iteration on first data seen
        if self._baseline_iter is None and state.rewards:
            self._baseline_iter = state.rewards[0][0]
            logger.info("Baseline iteration: %d", self._baseline_iter)
        if self._phase_start_iter is None and state.rewards:
            self._phase_start_iter = state.rewards[0][0]
            logger.info("Phase start iteration: %d", self._phase_start_iter)

        # Overfitting check — pass phase_start_iter for relative iteration gating
        reason = self._detector.check(
            state,
            phase_start_iter=self._phase_start_iter or 0,
        )
        if reason and not state.overfitting_detected:
            state.overfitting_detected = True
            state.overfitting_reason = reason
            logger.warning("Overfitting detected: %s", reason)
            if self._on_overfitting and not self._overfitting_fired:
                self._overfitting_fired = True
                self._on_overfitting(state)

        # Build summary
        if not state.rewards:
            return {"status": "NO_DATA"}

        best_ckpt_path = self.get_best_checkpoint()
        return {
            "status": "OVERFITTING" if state.overfitting_detected else "HEALTHY",
            "latest_iter": state.rewards[-1][0],
            "latest_reward": state.rewards[-1][1],
            "peak_reward": state.peak_reward,
            "best_model_iter": state.best_model_iter,
            "best_model_reward": state.best_model_reward,
            "best_checkpoint_path": str(best_ckpt_path) if best_ckpt_path else None,
            "reason": state.overfitting_reason,
        }

    # -- Accessors ----------------------------------------------------------- #

    def get_best_checkpoint(self) -> Optional[Path]:
        """Return the :class:`Path` to the best model checkpoint, or *None*."""
        if self._state is None or not self._state.rewards:
            return None
        return CheckpointAnalyzer.resolve_checkpoint(
            self._state.run_dir,
            self._state.best_model_iter,
        )

    @property
    def state(self) -> Optional[RunState]:
        """The underlying :class:`RunState` (read-only accessor)."""
        return self._state
