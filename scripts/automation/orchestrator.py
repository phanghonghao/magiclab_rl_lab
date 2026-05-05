#!/usr/bin/env python3
"""Training Orchestrator — automated multi-stage RL training pipeline.

Manages the full lifecycle of sequential training stages:
  - Swap environment configs between stages
  - Launch ``train.py`` as a subprocess
  - Monitor for overfitting via an embedded monitor
  - Gracefully stop training and advance to the next stage
  - Persist state for crash recovery

Usage::

    # Full plan, start from a specific stage
    python -m automation.orchestrator \\
        --plan training_plans/z1_5stage_plan.yaml \\
        --start-from s4_rough_l1

    # Dry run (print plan, don't execute)
    python -m automation.orchestrator \\
        --plan training_plans/z1_5stage_plan.yaml --dry-run

    # Ignore saved state, start fresh
    python -m automation.orchestrator \\
        --plan training_plans/z1_5stage_plan.yaml --fresh
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Ensure the parent ``scripts/`` directory is importable
_scripts_dir = str(Path(__file__).resolve().parent.parent)
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

# Ensure the magiclab_rl_lab root is importable for config_swapper defaults
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from automation.config_swapper import ConfigSwapper
from automation.embedded_monitor import EmbeddedMonitor
from automation.stage_manager import StageManager
from automation.state_store import OrchestratorState, StateStore
from automation.training_launcher import TrainingLauncher

logger = logging.getLogger("orchestrator")


# ──────────────────────────────────────────────────────────────────────────── #
# Orchestrator
# ──────────────────────────────────────────────────────────────────────────── #


class TrainingOrchestrator:
    """Main event loop that drives sequential training stages."""

    def __init__(
        self,
        plan_path: str,
        project_root: str = ".",
        state_path: str = "orchestrator_state.json",
        log_root: str = "logs/rsl_rl/magiclab_z1_12dof_velocity",
        poll_interval: int = 120,
        start_from: Optional[str] = None,
        fresh: bool = False,
        dry_run: bool = False,
        device: str = "cuda:0",
        num_gpus: int = 1,
    ):
        self._project_root = Path(project_root).resolve()
        self._poll_interval = poll_interval
        self._dry_run = dry_run
        self._device = device
        self._num_gpus = num_gpus

        # Components
        self._stage_mgr = StageManager(plan_path)
        self._state_store = StateStore(self._project_root / state_path)
        self._launcher = TrainingLauncher(
            train_script=str(self._project_root / "scripts" / "rsl_rl" / "train.py"),
            multigpu_script=str(self._project_root / "scripts" / "rsl_rl" / "train_multigpu.py"),
            log_dir=str(self._project_root / "logs"),
            cwd=str(self._project_root),
        )
        self._config_swapper = ConfigSwapper(project_root=str(self._project_root))
        self._log_root = self._project_root / log_root

        # Runtime state
        self._state: Optional[OrchestratorState] = None
        self._monitor: Optional[EmbeddedMonitor] = None
        self._proc = None  # subprocess.Popen

        # Resolve start stage
        if fresh:
            self._state_store.clear()
            self._start_stage_id = start_from or self._stage_mgr.get_start_stage_id()
        elif start_from:
            self._start_stage_id = start_from
        else:
            self._start_stage_id = self._stage_mgr.get_start_stage_id()

    # ── Public entry ────────────────────────────────────────────────────── #

    def run(self) -> None:
        """Main entry point."""
        self._setup_logging()

        if self._dry_run:
            self._print_dry_run()
            return

        # Attempt crash recovery
        self._state = self._state_store.load()
        if self._state and self._state.current_stage_status == "running":
            pid = self._state.training_pid
            if pid and TrainingLauncher.is_running(pid):
                logger.info(
                    "Recovering: stage '%s' still running (PID %d)",
                    self._state.current_stage_id,
                    pid,
                )
                self._resume_monitor()
            else:
                logger.info(
                    "State file found but PID %s is dead — marking as failed",
                    pid,
                )
                self._state.current_stage_status = "failed"

        if self._state is None or self._state.current_stage_status in ("complete", "failed"):
            self._init_new_run()

        # ── Main event loop ───────────────────────────────────────────── #
        logger.info("Orchestrator running (poll every %ds)", self._poll_interval)
        try:
            while True:
                status = self._state.current_stage_status
                if status == "pending":
                    self._start_stage()
                elif status == "running":
                    self._monitor_stage()
                elif status == "overfitting":
                    self._handle_overfitting()
                elif status == "complete":
                    self._advance_to_next_stage()
                elif status == "failed":
                    self._handle_failure()
                else:
                    logger.error("Unknown stage status: %s", status)
                    break

                self._state_store.save(self._state)
                time.sleep(self._poll_interval)
        except KeyboardInterrupt:
            logger.info("Interrupted — saving state and exiting")
            self._state_store.save(self._state)

    # ── Stage lifecycle ────────────────────────────────────────────────── #

    def _init_new_run(self) -> None:
        """Initialise a fresh orchestrator run (no prior state)."""
        self._state = OrchestratorState(
            plan_name=self._stage_mgr.plan_name,
            current_stage_id=self._start_stage_id,
            current_stage_status="pending",
            started_at=datetime.now().isoformat(),
        )
        self._state_store.save(self._state)
        logger.info(
            "New orchestration run: plan='%s', start='%s'",
            self._state.plan_name,
            self._start_stage_id,
        )

    def _start_stage(self) -> None:
        """Swap config, launch training, and set up monitoring."""
        stage_id = self._state.current_stage_id
        stage = self._stage_mgr.get_stage_by_id(stage_id)
        if stage is None:
            logger.error("Stage '%s' not found in plan", stage_id)
            self._state.current_stage_status = "failed"
            return

        logger.info("=== Starting stage: %s ===", stage_id)

        # 1) Swap env config
        env_cfg_path = self._project_root / stage.env_config
        if env_cfg_path.exists():
            self._config_swapper.swap(str(env_cfg_path), stage_id)
        else:
            logger.warning("Env config not found: %s — skipping swap", env_cfg_path)

        # 2) Resolve resume checkpoint
        checkpoint = self._stage_mgr.get_resume_checkpoint(
            stage_id, self._state.stage_history
        )

        # 3) Launch training
        run_name = stage_id
        self._proc = self._launcher.launch(
            run_name=run_name,
            max_iterations=stage.max_iterations,
            num_envs=stage.num_envs,
            device=self._device,
            checkpoint=checkpoint,
            num_gpus=self._num_gpus,
            extra_args=stage.extra_args,
        )

        # 4) Wait for Isaac Sim to initialise and create its run directory.
        #    Isaac Sim typically takes 60–120 s to start up.
        logger.info("Waiting for Isaac Sim to create run directory...")
        run_dir = None
        for attempt in range(8):  # 8 × 30 s = 4 min max
            time.sleep(30)
            run_dir = self._find_latest_run_dir(run_name)
            if run_dir is not None:
                break
            logger.info(
                "Run directory not found yet (attempt %d/8), waiting...",
                attempt + 1,
            )

        if run_dir is None:
            logger.error("Could not find run directory for '%s' after 4 min", stage_id)
            self._state.current_stage_status = "failed"
            return

        # 5) Set up embedded monitor
        monitor_cfg = stage.monitor or {}
        self._monitor = EmbeddedMonitor(
            log_root=str(self._log_root),
            terrain_type=stage.terrain,
            on_overfitting=self._on_overfitting_callback,
            action_rate_threshold=monitor_cfg.get("action_rate_threshold", -1.0),
            min_iterations=monitor_cfg.get("min_iterations", 2000),
        )
        self._monitor.start(run_dir)

        # 6) Update state
        self._state.training_pid = self._proc.pid
        self._state.training_run_dir = run_dir
        self._state.current_stage_status = "running"
        self._state.retry_count = 0
        logger.info(
            "Stage '%s' launched: PID=%d, run_dir=%s",
            stage_id,
            self._proc.pid,
            run_dir,
        )

    def _monitor_stage(self) -> None:
        """Poll the embedded monitor and check if the training process is alive."""
        pid = self._state.training_pid

        # Check if process is still alive
        if not TrainingLauncher.is_running(pid):
            # Process exited on its own — could be completion or crash
            returncode = self._proc.returncode if self._proc else -1
            if returncode == 0:
                logger.info("Training process exited normally (returncode=0)")
                # Treat as complete — run one final monitor poll
                if self._monitor:
                    self._monitor.poll()
                self._state.current_stage_status = "overfitting"  # will trigger best-checkpoint save
            else:
                logger.error("Training process exited (returncode=%s)", returncode)
                # Check if it hit max_iterations (log-based heuristic)
                if self._check_max_iterations_reached():
                    logger.info("Process exited after reaching max_iterations")
                    if self._monitor:
                        self._monitor.poll()
                    self._state.current_stage_status = "overfitting"
                else:
                    self._state.current_stage_status = "failed"
            return

        # Poll the monitor
        if self._monitor:
            summary = self._monitor.poll()
            status = summary.get("status", "NO_DATA")
            if status != "NO_DATA":
                logger.info(
                    "Stage '%s' — iter=%s, reward=%.2f, peak=%.2f, best=%s, %s",
                    self._state.current_stage_id,
                    summary.get("latest_iter", "?"),
                    summary.get("latest_reward", 0),
                    summary.get("peak_reward", 0),
                    summary.get("best_model_iter", "?"),
                    status,
                )

    def _handle_overfitting(self) -> None:
        """Kill training, save best checkpoint, and mark stage as complete."""
        stage_id = self._state.current_stage_id
        pid = self._state.training_pid

        logger.info("=== Handling overfitting for stage '%s' ===", stage_id)

        # 1) Stop training process
        if pid and TrainingLauncher.is_running(pid):
            self._launcher.graceful_stop(pid)

        # 2) Final monitor poll to get the best checkpoint
        best_ckpt = None
        best_reward = None
        if self._monitor:
            self._monitor.poll()
            ckpt_path = self._monitor.get_best_checkpoint()
            if ckpt_path:
                best_ckpt = str(ckpt_path)
                best_reward = self._monitor.state.best_model_reward if self._monitor.state else None
            elif self._monitor.state:
                # best_model_iter file might not be saved; try closest available
                best_iter = self._monitor.state.best_model_iter
                run_dir = Path(self._monitor.state.run_dir)
                for delta in range(0, best_iter + 1):
                    for candidate_iter in [best_iter - delta, best_iter + delta]:
                        ckpt_file = run_dir / f"model_{candidate_iter}.pt"
                        if ckpt_file.exists():
                            best_ckpt = str(ckpt_file)
                            break
                    if best_ckpt:
                        break

        if best_ckpt is None:
            # Fallback: find latest model file in run dir
            best_ckpt = self._find_latest_checkpoint()
            logger.warning("Using fallback checkpoint: %s", best_ckpt)

        # 3) Record in history
        self._state.stage_history.append({
            "stage_id": stage_id,
            "status": "overfitting",
            "best_checkpoint_path": best_ckpt,
            "best_reward": best_reward,
            "training_run_dir": self._state.training_run_dir,
            "completed_at": datetime.now().isoformat(),
        })
        self._state.best_checkpoint_path = best_ckpt
        self._state.best_reward = best_reward
        self._state.current_stage_status = "complete"

        logger.info(
            "Stage '%s' complete — best checkpoint: %s (reward: %s)",
            stage_id,
            best_ckpt,
            best_reward,
        )

    def _advance_to_next_stage(self) -> None:
        """Move to the next stage in the training plan."""
        current_id = self._state.current_stage_id
        next_stage = self._stage_mgr.get_next_stage(current_id)

        if next_stage is None:
            logger.info("=== All stages complete! ===")
            logger.info("Plan '%s' finished at %s", self._state.plan_name, datetime.now().isoformat())
            self._state_store.save(self._state)
            sys.exit(0)

        logger.info("Advancing: '%s' → '%s'", current_id, next_stage.id)
        self._state.current_stage_id = next_stage.id
        self._state.current_stage_status = "pending"
        self._state.training_pid = None
        self._state.training_run_dir = None
        self._state.retry_count = 0

    def _handle_failure(self) -> None:
        """Retry the current stage or give up."""
        stage_id = self._state.current_stage_id
        max_retries = self._stage_mgr.retry_policy.max_retries
        retry_count = self._state.retry_count

        if retry_count < max_retries:
            logger.warning(
                "Stage '%s' failed — retry %d/%d",
                stage_id,
                retry_count + 1,
                max_retries,
            )
            self._state.retry_count = retry_count + 1
            self._state.current_stage_status = "pending"
            self._state.training_pid = None
            self._state.training_run_dir = None
        else:
            logger.error(
                "Stage '%s' failed after %d retries — stopping orchestrator",
                stage_id,
                max_retries,
            )
            self._state_store.save(self._state)
            sys.exit(1)

    # ── Callbacks ──────────────────────────────────────────────────────── #

    def _on_overfitting_callback(self, run_state) -> None:
        """Callback from :class:`EmbeddedMonitor` when overfitting is detected."""
        logger.warning(
            "Overfitting callback fired for stage '%s': %s",
            self._state.current_stage_id,
            run_state.overfitting_reason,
        )
        self._state.current_stage_status = "overfitting"

    # ── Recovery ───────────────────────────────────────────────────────── #

    def _resume_monitor(self) -> None:
        """Re-attach an :class:`EmbeddedMonitor` to a running training process."""
        stage_id = self._state.current_stage_id
        stage = self._stage_mgr.get_stage_by_id(stage_id)
        terrain = stage.terrain if stage else "gentle"
        run_dir = self._state.training_run_dir

        if run_dir and Path(run_dir).exists():
            monitor_cfg = (stage.monitor or {}) if stage else {}
            self._monitor = EmbeddedMonitor(
                log_root=str(self._log_root),
                terrain_type=terrain,
                on_overfitting=self._on_overfitting_callback,
                action_rate_threshold=monitor_cfg.get("action_rate_threshold", -1.0),
                min_iterations=monitor_cfg.get("min_iterations", 2000),
            )
            self._monitor.start(run_dir)
            logger.info("Monitor re-attached to running stage '%s'", stage_id)
        else:
            logger.warning("Run dir '%s' not found — cannot re-attach monitor", run_dir)

    # ── Helpers ────────────────────────────────────────────────────────── #

    def _find_latest_run_dir(self, run_name: str) -> Optional[str]:
        """Find the run directory for the just-launched training.

        train.py creates directories named ``YYYY-MM-DD_HH-MM-SS_{run_name}``
        when ``--run_name`` is passed.  We filter by that suffix and pick the
        newest match.  Returns *None* if no matching directory exists yet.
        """
        if not self._log_root.exists():
            return None

        suffix = f"_{run_name}"
        candidates = [
            d for d in self._log_root.iterdir()
            if d.is_dir() and not d.name.startswith(".") and d.name.endswith(suffix)
        ]

        if not candidates:
            return None

        # Pick the most recently created
        dirs = sorted(candidates, key=lambda d: d.stat().st_ctime, reverse=True)
        logger.info("Found run directory: %s", dirs[0].name)
        return str(dirs[0])

    def _find_latest_checkpoint(self) -> Optional[str]:
        """Fallback: find the latest ``model_*.pt`` in the run directory."""
        run_dir = self._state.training_run_dir
        if not run_dir:
            return None
        p = Path(run_dir)
        ckpts = sorted(p.glob("model_*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
        return str(ckpts[0]) if ckpts else None

    def _check_max_iterations_reached(self) -> bool:
        """Heuristic: check if the training log shows it reached max iterations."""
        stage_id = self._state.current_stage_id
        stage = self._stage_mgr.get_stage_by_id(stage_id)
        if stage is None:
            return False

        log_file = self._project_root / "logs" / f"train_{stage_id}.log"
        if not log_file.exists():
            return False

        try:
            # Read the last few KB and look for completion indicators
            with open(log_file, "r", encoding="utf-8", errors="ignore") as fh:
                fh.seek(max(0, log_file.stat().st_size - 5000))
                tail = fh.read()
            # Check for iteration count near max
            max_iter = stage.max_iterations
            for line in reversed(tail.splitlines()):
                if f"{max_iter}/{max_iter}" in line or f"/{max_iter}" in line:
                    return True
        except Exception:
            pass
        return False

    # ── Logging setup ──────────────────────────────────────────────────── #

    def _setup_logging(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(
                    self._project_root / "logs" / "orchestrator.log",
                    mode="a",
                    encoding="utf-8",
                ),
            ],
        )

    # ── Dry run ────────────────────────────────────────────────────────── #

    def _print_dry_run(self) -> None:
        """Print the plan without executing anything."""
        print(f"Training Plan: {self._stage_mgr.plan_name}")
        print(f"Project Root:  {self._project_root}")
        print(f"Log Root:      {self._log_root}")
        print(f"Start From:    {self._start_stage_id}")
        print()

        start_idx = self._stage_mgr.get_stage_index(self._start_stage_id)
        for i, stage in enumerate(self._stage_mgr.stages):
            marker = ">>>" if i >= start_idx else "   "
            resume = f" (resume from {stage.resume_from})" if stage.resume_from else ""
            ckpt_label = ""
            if stage.initial_checkpoint:
                ckpt_label = f" (checkpoint: {stage.initial_checkpoint})"
            print(f"  {marker} [{stage.id}]")
            print(f"       terrain:         {stage.terrain}")
            print(f"       env_config:      {stage.env_config}")
            print(f"       max_iterations:  {stage.max_iterations}")
            print(f"       num_envs:        {stage.num_envs}{resume}{ckpt_label}")
            if stage.monitor:
                print(f"       monitor:         {stage.monitor}")

            # Show the command that would be run
            if i >= start_idx:
                ckpt_flag = ""
                if stage.resume_from:
                    ckpt_flag = f" --checkpoint <best_from_{stage.resume_from}>"
                elif stage.initial_checkpoint:
                    ckpt_flag = f" --checkpoint {stage.initial_checkpoint}"

                if self._num_gpus > 1:
                    ckpt_resume = ""
                    if stage.initial_checkpoint:
                        ckpt_path = Path(stage.initial_checkpoint)
                        ckpt_resume = f" --resume --load_run={ckpt_path.parent.name} --checkpoint={ckpt_path.name}"
                    cmd = (
                        f"torchrun --nproc_per_node={self._num_gpus} "
                        f"scripts/rsl_rl/train_multigpu.py"
                        f" --task=Magiclab-Z1-12dof-Velocity --headless --distributed"
                        f" --num_envs={stage.num_envs}"
                        f" --max_iterations={stage.max_iterations}"
                        f" --run_name={stage.id}"
                        f"{ckpt_resume}"
                    )
                else:
                    cmd = (
                        f"python -u scripts/rsl_rl/train.py"
                        f" --task Magiclab-Z1-12dof-Velocity --headless"
                        f" --device {self._device}"
                        f" --num_envs {stage.num_envs}"
                        f" --max_iterations {stage.max_iterations}"
                        f" --run_name {stage.id}"
                        f"{ckpt_flag}"
                    )
                print(f"       command: {cmd}")
            print()

        rp = self._stage_mgr.retry_policy
        print(f"  Retry policy: max_retries={rp.max_retries}")
        if rp.nan:
            print(f"                 on NaN: {rp.nan}")
        if rp.oom:
            print(f"                 on OOM: {rp.oom}")
        print()
        print("Dry run — no training will be launched.")


# ──────────────────────────────────────────────────────────────────────────── #
# CLI
# ──────────────────────────────────────────────────────────────────────────── #


def parse_args():
    parser = argparse.ArgumentParser(
        description="Training Orchestrator — automated multi-stage RL training",
    )
    parser.add_argument(
        "--plan",
        type=str,
        required=True,
        help="Path to the YAML training plan file",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=".",
        help="Root directory of magiclab_rl_lab (default: current directory)",
    )
    parser.add_argument(
        "--log-root",
        type=str,
        default="logs/rsl_rl/magiclab_z1_12dof_velocity",
        help="Relative path to the training log root",
    )
    parser.add_argument(
        "--start-from",
        type=str,
        default=None,
        help="Stage ID to start from (default: first stage in plan)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore saved state and start fresh",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan and commands without executing",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=120,
        help="Seconds between monitoring polls (default: 120)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="CUDA device for training (default: cuda:0)",
    )
    parser.add_argument(
        "--state-file",
        type=str,
        default="orchestrator_state.json",
        help="Filename for crash-recovery state (default: orchestrator_state.json)",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs for distributed training (default: 1, single-GPU)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    orchestrator = TrainingOrchestrator(
        plan_path=args.plan,
        project_root=args.project_root,
        state_path=args.state_file,
        log_root=args.log_root,
        poll_interval=args.poll_interval,
        start_from=args.start_from,
        fresh=args.fresh,
        dry_run=args.dry_run,
        device=args.device,
        num_gpus=args.num_gpus,
    )
    orchestrator.run()


if __name__ == "__main__":
    main()
