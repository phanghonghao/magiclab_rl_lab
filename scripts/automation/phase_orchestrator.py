#!/usr/bin/env python3
"""Phase-Based Training Orchestrator — 5-phase automated RL pipeline.

Two-level event loop:
  Outer: iterate over phases (p1 → p5)
  Inner: iterate over sub-phases within each phase (coarse → fine)

Each sub-phase:
  1. config_generator generates velocity_env_cfg.py
  2. ppo_override generates a temporary PPO config
  3. Training launched on N GPUs via torchrun
  4. embedded_monitor watches for overfitting
  5. On overfitting: stop, save best checkpoint
  6. Rollback check: if best_reward < starting_reward * 0.95 → retry with LR×0.5
  7. Record simulation video
  8. Advance to next sub-phase (or next phase)

Usage::

    # Full pipeline
    python -m automation.phase_orchestrator \\
        --plan training_plans/z1_5phase_plan.yaml \\
        --num-gpus 4

    # Dry run
    python -m automation.phase_orchestrator \\
        --plan training_plans/z1_5phase_plan.yaml --dry-run

    # Smoke test (verify pipeline with minimal iterations)
    python -m automation.phase_orchestrator \\
        --plan training_plans/z1_5phase_plan.yaml --smoke-test --num-gpus 4

    # Start from specific sub-phase
    python -m automation.phase_orchestrator \\
        --plan training_plans/z1_5phase_plan.yaml \\
        --start-from p3_coarse --num-gpus 4

    # Fresh start (ignore saved state)
    python -m automation.phase_orchestrator \\
        --plan training_plans/z1_5phase_plan.yaml --fresh --num-gpus 4
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Ensure the parent ``scripts/`` directory is importable
_scripts_dir = str(Path(__file__).resolve().parent.parent)
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from automation.config_generator import generate_env_config, _ACTIVE_CFG_REL
from automation.embedded_monitor import EmbeddedMonitor
from automation.phase_manager import PhaseManager, SubPhaseConfig
from automation.ppo_override import generate_ppo_override
from automation.state_store import OrchestratorState, StateStore
from automation.training_launcher import TrainingLauncher

logger = logging.getLogger("phase_orchestrator")


class PhaseOrchestrator:
    """Main orchestrator for the 5-phase automated training pipeline."""

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
        smoke_test: bool = False,
        num_gpus: int = 4,
    ):
        self._project_root = Path(project_root).resolve()
        self._poll_interval = poll_interval
        self._dry_run = dry_run
        self._smoke_test = smoke_test
        self._num_gpus = num_gpus

        # Components
        self._phase_mgr = PhaseManager(plan_path)
        self._state_store = StateStore(self._project_root / state_path)
        self._launcher = TrainingLauncher(
            train_script=str(self._project_root / "scripts" / "rsl_rl" / "train.py"),
            multigpu_script=str(self._project_root / "scripts" / "rsl_rl" / "train_multigpu.py"),
            log_dir=str(self._project_root / "logs"),
            cwd=str(self._project_root),
        )
        self._log_root = self._project_root / log_root

        # Paths for generated configs
        self._tmp_dir = self._project_root / "tmp" / "phase_configs"
        self._active_env_cfg = self._project_root / _ACTIVE_CFG_REL

        # Runtime state
        self._state: Optional[OrchestratorState] = None
        self._monitor: Optional[EmbeddedMonitor] = None
        self._proc = None

        # Resolve start sub-phase
        if fresh:
            self._state_store.clear()
            self._start_id = start_from or self._phase_mgr.get_start_sub_phase_id()
        elif start_from:
            self._start_id = start_from
        else:
            self._start_id = self._phase_mgr.get_start_sub_phase_id()

    # ── Public entry ────────────────────────────────────────────── #

    def run(self) -> None:
        """Main entry point."""
        self._setup_logging()

        if self._dry_run:
            self._print_dry_run()
            return

        if self._smoke_test:
            self._run_smoke_test()
            return

        # Crash recovery
        self._state = self._state_store.load()
        if self._state and self._state.current_stage_status == "running":
            pid = self._state.training_pid
            if pid and TrainingLauncher.is_running(pid):
                logger.info("Recovering: sub-phase '%s' still running (PID %d)",
                            self._state.current_stage_id, pid)
                self._resume_monitor()
            else:
                logger.info("State found but PID %s is dead — marking as failed", pid)
                self._state.current_stage_status = "failed"

        if self._state is None or self._state.current_stage_status in ("complete", "failed"):
            self._init_new_run()

        # Main event loop
        logger.info("Phase Orchestrator running (poll every %ds, %d GPUs)",
                     self._poll_interval, self._num_gpus)
        try:
            while True:
                status = self._state.current_stage_status
                if status == "pending":
                    self._start_sub_phase()
                elif status == "running":
                    self._monitor_sub_phase()
                elif status == "overfitting":
                    self._handle_overfitting()
                elif status == "complete":
                    self._advance()
                elif status == "failed":
                    self._handle_failure()
                else:
                    logger.error("Unknown status: %s", status)
                    break

                self._state_store.save(self._state)
                time.sleep(self._poll_interval)
        except KeyboardInterrupt:
            logger.info("Interrupted — saving state and exiting")
            self._state_store.save(self._state)

    # ── Sub-phase lifecycle ─────────────────────────────────────── #

    def _run_smoke_test(self) -> None:
        """Run every sub-phase with minimal iterations to verify the full pipeline."""
        logger.info("=" * 60)
        logger.info("=== SMOKE TEST START ===")
        logger.info("=" * 60)
        logger.info("Plan: %s, GPUs: %d", self._phase_mgr.plan_name, self._num_gpus)

        all_sub_phases = self._phase_mgr.all_sub_phases
        logger.info("Sub-phases to verify: %s", [sp.id for sp in all_sub_phases])

        # Initialize a transient state for checkpoint chaining
        self._state = OrchestratorState(
            plan_name=self._phase_mgr.plan_name,
            current_stage_id=all_sub_phases[0].id,
            current_stage_status="pending",
            started_at=datetime.now().isoformat(),
        )

        all_passed = True
        smoke_runs: list[str] = []  # collect all smoke run dirs for cleanup

        for idx, sp in enumerate(all_sub_phases):
            logger.info("-" * 50)
            logger.info("Smoke sub-phase [%d/%d]: %s (%s)",
                         idx + 1, len(all_sub_phases), sp.id, sp.name)
            ok, run_dir = self._smoke_run_sub_phase(sp)
            if not ok:
                all_passed = False
                break
            if run_dir:
                smoke_runs.append(run_dir)

        logger.info("=" * 60)
        if all_passed:
            self._cleanup_smoke_runs(smoke_runs)
            logger.info("=== SMOKE TEST PASSED ===")
            logger.info("All %d sub-phases verified OK", len(all_sub_phases))
            logger.info("Safe to launch full pipeline with --fresh")
            sys.exit(0)
        else:
            logger.error("=== SMOKE TEST FAILED ===")
            if smoke_runs:
                logger.error("Smoke run directories preserved for diagnosis:")
                for r in smoke_runs:
                    logger.error("  %s", r)
            sys.exit(1)

    def _smoke_run_sub_phase(self, sp: SubPhaseConfig) -> tuple[bool, Optional[str]]:
        """Run one sub-phase with minimal iterations. Returns (success, run_dir)."""
        logger.info("--- Smoke: %s (%s) ---", sp.id, sp.name)

        # 1) Generate + swap env config
        merged_params = {"env": sp.env, "rewards": sp.rewards}
        env_cfg_path = self._tmp_dir / sp.id / "velocity_env_cfg.py"
        generate_env_config(merged_params, env_cfg_path, project_root=self._project_root)
        self._swap_active_config(env_cfg_path, sp.id)

        # 2) Generate PPO override
        default_ppo = self._phase_mgr.defaults.get("ppo", {})
        ppo_cfg_path = self._tmp_dir / sp.id / "ppo_override_cfg.py"
        generate_ppo_override(sp.ppo, default_ppo, ppo_cfg_path)

        # 3) Resolve checkpoint (chains from previous smoke sub-phase)
        checkpoint = self._resolve_checkpoint(sp.id)

        # 4) Launch with minimal params
        run_name = f"smoke_{sp.id}"
        proc = self._launcher.launch(
            run_name=run_name,
            max_iterations=50,
            num_envs=64,
            device="cuda:0",
            task=self._phase_mgr.task,
            checkpoint=checkpoint,
            num_gpus=self._num_gpus,
            agent_cfg=str(ppo_cfg_path),
        )

        # 5) Wait for run directory (up to 5 min)
        run_dir = None
        for attempt in range(10):
            time.sleep(30)
            run_dir = self._find_latest_run_dir(run_name)
            if run_dir:
                break
            logger.info("  Waiting for run dir... (%d/10)", attempt + 1)

        if not run_dir:
            logger.error("FAIL: %s — run directory not created within 5 min", sp.id)
            self._launcher.graceful_stop(proc.pid)
            return False, None

        logger.info("  Run dir created: %s", run_dir)

        # 6) Wait for at least 1 checkpoint (up to 3 min)
        ckpt_found = False
        for attempt in range(18):
            time.sleep(10)
            ckpts = list(Path(run_dir).glob("model_*.pt"))
            if ckpts:
                ckpt_found = True
                break
            logger.info("  Waiting for checkpoint... (%d/18)", attempt + 1)

        # 7) Stop training
        if TrainingLauncher.is_running(proc.pid):
            self._launcher.graceful_stop(proc.pid)

        if not ckpt_found:
            logger.error("FAIL: %s — no checkpoint saved within 3 min", sp.id)
            return False, run_dir

        # 8) Record in state history for next sub-phase's checkpoint resolution
        latest_ckpt = sorted(
            Path(run_dir).glob("model_*.pt"),
            key=lambda x: x.stat().st_mtime, reverse=True,
        )[0]
        self._state.stage_history.append({
            "sub_phase_id": sp.id,
            "best_checkpoint_path": str(latest_ckpt),
            "best_reward": 0.0,
            "training_run_dir": run_dir,
        })

        logger.info("PASS: %s (checkpoint: %s)", sp.id, latest_ckpt.name)
        return True, run_dir

    def _cleanup_smoke_runs(self, smoke_runs: list[str]) -> None:
        """Remove all smoke test run directories."""
        for run_dir_str in smoke_runs:
            run_dir = Path(run_dir_str)
            if run_dir.exists() and "smoke_" in run_dir.name:
                try:
                    shutil.rmtree(run_dir)
                    logger.info("Cleaned up smoke run: %s", run_dir)
                except Exception as exc:
                    logger.warning("Failed to clean up %s: %s", run_dir, exc)

    # ── Sub-phase lifecycle ─────────────────────────────────────── #

    def _init_new_run(self) -> None:
        self._state = OrchestratorState(
            plan_name=self._phase_mgr.plan_name,
            current_stage_id=self._start_id,
            current_phase_id=self._get_phase_id(self._start_id),
            current_stage_status="pending",
            started_at=datetime.now().isoformat(),
        )
        self._state_store.save(self._state)
        logger.info("New orchestration run: plan='%s', start='%s'",
                     self._state.plan_name, self._start_id)

    def _start_sub_phase(self) -> None:
        """Generate configs, swap env, launch training."""
        sp_id = self._state.current_stage_id
        sp = self._phase_mgr.get_sub_phase(sp_id)
        if sp is None:
            logger.error("Sub-phase '%s' not found in plan", sp_id)
            self._state.current_stage_status = "failed"
            return

        logger.info("=== Starting sub-phase: %s (%s) ===", sp_id, sp.name)

        # 1) Generate env config
        merged_params = {"env": sp.env, "rewards": sp.rewards}
        env_cfg_path = self._tmp_dir / sp_id / "velocity_env_cfg.py"
        generate_env_config(merged_params, env_cfg_path)

        # 2) Swap active env config
        self._swap_active_config(env_cfg_path, sp_id)

        # 3) Generate PPO override config
        default_ppo = self._phase_mgr.defaults.get("ppo", {})
        ppo_cfg_path = self._tmp_dir / sp_id / "ppo_override_cfg.py"
        generate_ppo_override(sp.ppo, default_ppo, ppo_cfg_path)

        # 4) Resolve checkpoint
        checkpoint = self._resolve_checkpoint(sp_id)

        # 5) Launch training
        run_name = sp_id
        self._proc = self._launcher.launch(
            run_name=run_name,
            max_iterations=sp.max_iterations,
            num_envs=sp.num_envs,
            device="cuda:0",
            task=self._phase_mgr.task,
            checkpoint=checkpoint,
            num_gpus=self._num_gpus,
            agent_cfg=str(ppo_cfg_path),
        )

        # 6) Wait for Isaac Sim to create run directory
        logger.info("Waiting for Isaac Sim to create run directory...")
        run_dir = None
        for attempt in range(8):
            time.sleep(30)
            run_dir = self._find_latest_run_dir(run_name)
            if run_dir is not None:
                break
            logger.info("Run directory not found (attempt %d/8)...", attempt + 1)

        if run_dir is None:
            logger.error("Could not find run directory for '%s' after 4 min", sp_id)
            self._state.current_stage_status = "failed"
            return

        # 7) Setup monitor
        monitor_cfg = sp.monitor
        self._monitor = EmbeddedMonitor(
            log_root=str(self._log_root),
            terrain_type=sp.terrain,
            on_overfitting=self._on_overfitting_callback,
            action_rate_threshold=monitor_cfg.get("action_rate_threshold", -1.0),
            min_iterations=monitor_cfg.get("min_iterations", 2000),
        )
        self._monitor.start(run_dir)

        # 8) Update state
        self._state.training_pid = self._proc.pid
        self._state.training_run_dir = run_dir
        self._state.current_stage_status = "running"
        self._state.current_phase_id = sp.phase_id
        self._state.rollback_count = 0

        # Record starting reward for rollback check
        if self._state.starting_reward is None:
            self._state.starting_reward = self._state.best_reward

        logger.info("Sub-phase '%s' launched: PID=%d, run_dir=%s", sp_id, self._proc.pid, run_dir)

    def _monitor_sub_phase(self) -> None:
        """Poll monitor and check process liveness."""
        pid = self._state.training_pid

        if not TrainingLauncher.is_running(pid):
            returncode = self._proc.returncode if self._proc else -1
            if returncode == 0:
                logger.info("Training process exited normally")
                if self._monitor:
                    self._monitor.poll()
                self._state.current_stage_status = "overfitting"
            else:
                logger.error("Training process exited (returncode=%s)", returncode)
                if self._check_max_iterations_reached():
                    logger.info("Process exited after reaching max_iterations")
                    if self._monitor:
                        self._monitor.poll()
                    self._state.current_stage_status = "overfitting"
                else:
                    self._state.current_stage_status = "failed"
            return

        if self._monitor:
            summary = self._monitor.poll()
            status = summary.get("status", "NO_DATA")
            if status != "NO_DATA":
                logger.info(
                    "[%s] iter=%s, reward=%.2f, peak=%.2f, best=%s, %s",
                    self._state.current_stage_id,
                    summary.get("latest_iter", "?"),
                    summary.get("latest_reward", 0),
                    summary.get("peak_reward", 0),
                    summary.get("best_model_iter", "?"),
                    status,
                )

    def _handle_overfitting(self) -> None:
        """Stop training, save best checkpoint, check rollback."""
        sp_id = self._state.current_stage_id
        pid = self._state.training_pid

        logger.info("=== Handling overfitting for '%s' ===", sp_id)

        # 1) Stop training
        if pid and TrainingLauncher.is_running(pid):
            self._launcher.graceful_stop(pid)

        # 2) Final monitor poll
        best_ckpt = None
        best_reward = None
        if self._monitor:
            self._monitor.poll()
            ckpt_path = self._monitor.get_best_checkpoint()
            if ckpt_path:
                best_ckpt = str(ckpt_path)
                best_reward = self._monitor.state.best_model_reward if self._monitor.state else None
            elif self._monitor.state:
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
            best_ckpt = self._find_latest_checkpoint()
            logger.warning("Using fallback checkpoint: %s", best_ckpt)

        # 3) Rollback check
        sp = self._phase_mgr.get_sub_phase(sp_id)
        rollback_cfg = sp.rollback if sp else {}
        reward_threshold = rollback_cfg.get("reward_threshold", 0.95)
        max_retries = rollback_cfg.get("max_retries", 1)
        starting_reward = self._state.starting_reward

        should_rollback = False
        if starting_reward is not None and starting_reward > 0 and best_reward is not None:
            if best_reward < starting_reward * reward_threshold:
                should_rollback = True
                logger.warning(
                    "ROLLBACK: best_reward=%.2f < starting_reward=%.2f × %.2f",
                    best_reward, starting_reward, reward_threshold,
                )

        if should_rollback and self._state.rollback_count < max_retries:
            # Retry with reduced LR
            self._state.rollback_count += 1
            logger.info("Retrying '%s' (attempt %d/%d) with LR×0.5",
                         sp_id, self._state.rollback_count, max_retries)
            self._reduce_lr_in_override(sp_id)
            self._state.current_stage_status = "pending"
            self._state.training_pid = None
            self._state.training_run_dir = None
            return

        # 4) Record in history
        self._state.stage_history.append({
            "sub_phase_id": sp_id,
            "phase_id": self._state.current_phase_id,
            "status": "overfitting" if not should_rollback else "rollback_exhausted",
            "best_checkpoint_path": best_ckpt,
            "best_reward": best_reward,
            "starting_reward": starting_reward,
            "rollback_count": self._state.rollback_count,
            "training_run_dir": self._state.training_run_dir,
            "completed_at": datetime.now().isoformat(),
        })
        self._state.best_checkpoint_path = best_ckpt
        self._state.best_reward = best_reward
        self._state.starting_reward = best_reward  # for next sub-phase

        # 5) Record video
        self._record_video(sp_id, best_ckpt)

        # 6) Check if this is the last sub-phase of a phase
        self._check_phase_completion()

        self._state.current_stage_status = "complete"

        logger.info("Sub-phase '%s' complete — best: %s (reward: %s)",
                     sp_id, best_ckpt, best_reward)

    def _advance(self) -> None:
        """Move to next sub-phase or finish."""
        current_id = self._state.current_stage_id
        next_sp = self._phase_mgr.get_next_sub_phase(current_id)

        if next_sp is None:
            logger.info("=== ALL SUB-PHASES COMPLETE! ===")
            logger.info("Plan '%s' finished at %s",
                         self._state.plan_name, datetime.now().isoformat())
            self._state_store.save(self._state)
            sys.exit(0)

        logger.info("Advancing: '%s' → '%s'", current_id, next_sp.id)
        self._state.current_stage_id = next_sp.id
        self._state.current_phase_id = next_sp.phase_id
        self._state.current_stage_status = "pending"
        self._state.training_pid = None
        self._state.training_run_dir = None
        self._state.rollback_count = 0

    def _handle_failure(self) -> None:
        """Retry or give up."""
        sp_id = self._state.current_stage_id
        max_retries = 2

        if self._state.retry_count < max_retries:
            self._state.retry_count += 1
            logger.warning("Sub-phase '%s' failed — retry %d/%d",
                           sp_id, self._state.retry_count, max_retries)
            self._state.current_stage_status = "pending"
            self._state.training_pid = None
            self._state.training_run_dir = None
        else:
            logger.error("Sub-phase '%s' failed after %d retries — stopping", sp_id, max_retries)
            self._state_store.save(self._state)
            sys.exit(1)

    # ── Rollback helpers ────────────────────────────────────────── #

    def _reduce_lr_in_override(self, sp_id: str) -> None:
        """Regenerate the PPO override with LR×0.5 for rollback retry."""
        sp = self._phase_mgr.get_sub_phase(sp_id)
        if sp is None:
            return
        current_lr = sp.ppo.get("learning_rate", 1e-3)
        # Reduce LR by half, respecting any prior reductions
        new_lr = current_lr * (0.5 ** self._state.rollback_count)
        sp.ppo["learning_rate"] = new_lr

        default_ppo = self._phase_mgr.defaults.get("ppo", {})
        ppo_cfg_path = self._tmp_dir / sp_id / "ppo_override_cfg.py"
        generate_ppo_override(sp.ppo, default_ppo, ppo_cfg_path)
        logger.info("Reduced LR to %.2e for rollback retry", new_lr)

    # ── Phase completion ────────────────────────────────────────── #

    def _check_phase_completion(self) -> None:
        """Record phase completion if the current sub-phase is the last in its phase."""
        sp_id = self._state.current_stage_id
        phase = self._phase_mgr.get_phase_for_sub_phase(sp_id)
        if phase is None:
            return

        last_sp_in_phase = phase.sub_phases[-1]
        if sp_id == last_sp_in_phase.id:
            self._state.phase_history.append({
                "phase_id": phase.id,
                "phase_name": phase.name,
                "completed_at": datetime.now().isoformat(),
                "best_checkpoint_path": self._state.best_checkpoint_path,
                "best_reward": self._state.best_reward,
            })
            logger.info("=== Phase '%s' (%s) COMPLETE ===", phase.id, phase.name)

    # ── Video recording ─────────────────────────────────────────── #

    def _record_video(self, sp_id: str, checkpoint: Optional[str]) -> None:
        """Record a simulation video using the play script."""
        if not checkpoint:
            logger.warning("No checkpoint for video recording of '%s'", sp_id)
            return

        video_dir = self._project_root / "videos" / "phase_pipeline"
        video_dir.mkdir(parents=True, exist_ok=True)

        play_script = self._project_root / "scripts" / "rsl_rl" / "play.py"
        if not play_script.exists():
            logger.warning("play.py not found — skipping video recording")
            return

        ckpt_path = Path(checkpoint)
        run_dir = ckpt_path.parent
        ckpt_name = ckpt_path.name

        video_file = video_dir / f"{sp_id}.mp4"

        logger.info("Recording video for '%s'...", sp_id)
        try:
            cmd = [
                "python", "-u", str(play_script),
                f"--task={self._phase_mgr.task}",
                "--headless",
                "--video",
                "--video_length=400",
                f"--load_run={run_dir.name}",
                f"--checkpoint={ckpt_name}",
                "--num_envs=16",
            ]
            result = subprocess.run(
                cmd,
                cwd=str(self._project_root),
                capture_output=True,
                text=True,
                timeout=600,
            )
            # Find the generated video and rename
            _find_and_rename_video(run_dir, video_file)
            logger.info("Video recorded: %s", video_file)
        except subprocess.TimeoutExpired:
            logger.warning("Video recording timed out for '%s'", sp_id)
        except Exception as exc:
            logger.warning("Video recording failed for '%s': %s", sp_id, exc)

    # ── Config helpers ──────────────────────────────────────────── #

    def _swap_active_config(self, new_cfg_path: Path, sp_id: str) -> None:
        """Backup and replace the active velocity_env_cfg.py."""
        if not self._active_env_cfg.exists():
            logger.warning("Active env config not found: %s", self._active_env_cfg)
            return

        # Backup
        backup = self._active_env_cfg.parent / f"{self._active_env_cfg.name}.bak.{sp_id}"
        shutil.copy2(str(self._active_env_cfg), str(backup))

        # Replace
        shutil.copy2(str(new_cfg_path), str(self._active_env_cfg))
        logger.info("Swapped env config for '%s'", sp_id)

    def _resolve_checkpoint(self, sp_id: str) -> Optional[str]:
        """Resolve the checkpoint to resume from."""
        # Check history for the previous sub-phase
        if not self._state.stage_history:
            return None

        # Use the latest completed sub-phase's best checkpoint
        for entry in reversed(self._state.stage_history):
            ckpt = entry.get("best_checkpoint_path")
            if ckpt and Path(ckpt).exists():
                logger.info("Resuming from: %s", ckpt)
                return ckpt

        return None

    # ── Callbacks ───────────────────────────────────────────────── #

    def _on_overfitting_callback(self, run_state) -> None:
        logger.warning("Overfitting detected for '%s': %s",
                        self._state.current_stage_id, run_state.overfitting_reason)
        self._state.current_stage_status = "overfitting"

    # ── Recovery ────────────────────────────────────────────────── #

    def _resume_monitor(self) -> None:
        sp_id = self._state.current_stage_id
        sp = self._phase_mgr.get_sub_phase(sp_id)
        terrain = sp.terrain if sp else "gentle"
        run_dir = self._state.training_run_dir

        if run_dir and Path(run_dir).exists():
            monitor_cfg = sp.monitor if sp else {}
            self._monitor = EmbeddedMonitor(
                log_root=str(self._log_root),
                terrain_type=terrain,
                on_overfitting=self._on_overfitting_callback,
                action_rate_threshold=monitor_cfg.get("action_rate_threshold", -1.0),
                min_iterations=monitor_cfg.get("min_iterations", 2000),
            )
            self._monitor.start(run_dir)
            logger.info("Monitor re-attached to '%s'", sp_id)
        else:
            logger.warning("Run dir '%s' not found — cannot re-attach monitor", run_dir)

    # ── Helpers ─────────────────────────────────────────────────── #

    def _get_phase_id(self, sp_id: str) -> str:
        phase = self._phase_mgr.get_phase_for_sub_phase(sp_id)
        return phase.id if phase else ""

    def _find_latest_run_dir(self, run_name: str) -> Optional[str]:
        if not self._log_root.exists():
            return None

        suffix = f"_{run_name}"
        candidates = [
            d for d in self._log_root.iterdir()
            if d.is_dir() and not d.name.startswith(".") and d.name.endswith(suffix)
        ]
        if not candidates:
            return None

        dirs = sorted(candidates, key=lambda d: d.stat().st_ctime, reverse=True)
        return str(dirs[0])

    def _find_latest_checkpoint(self) -> Optional[str]:
        run_dir = self._state.training_run_dir
        if not run_dir:
            return None
        p = Path(run_dir)
        ckpts = sorted(p.glob("model_*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
        return str(ckpts[0]) if ckpts else None

    def _check_max_iterations_reached(self) -> bool:
        sp_id = self._state.current_stage_id
        sp = self._phase_mgr.get_sub_phase(sp_id)
        if sp is None:
            return False

        log_file = self._project_root / "logs" / f"train_{sp_id}.log"
        if not log_file.exists():
            return False

        try:
            with open(log_file, "r", encoding="utf-8", errors="ignore") as fh:
                fh.seek(max(0, log_file.stat().st_size - 5000))
                tail = fh.read()
            max_iter = sp.max_iterations
            for line in reversed(tail.splitlines()):
                if f"{max_iter}/{max_iter}" in line or f"/{max_iter}" in line:
                    return True
        except Exception:
            pass
        return False

    # ── Logging ─────────────────────────────────────────────────── #

    def _setup_logging(self) -> None:
        log_dir = self._project_root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(log_dir / "phase_orchestrator.log", mode="a", encoding="utf-8"),
            ],
        )

    # ── Dry run ─────────────────────────────────────────────────── #

    def _print_dry_run(self) -> None:
        print(f"Training Plan: {self._phase_mgr.plan_name}")
        print(f"Project Root:  {self._project_root}")
        print(f"Log Root:      {self._log_root}")
        print(f"GPUs:          {self._num_gpus}")
        print(f"Start From:    {self._start_id}")
        print()

        start_idx = self._phase_mgr.get_sub_phase_index(self._start_id)

        for phase in self._phase_mgr.phases:
            print(f"  Phase [{phase.id}] {phase.name}  (terrain: {phase.terrain})")
            for sp in phase.sub_phases:
                idx = self._phase_mgr.get_sub_phase_index(sp.id)
                marker = ">>>" if idx >= start_idx else "   "
                print(f"    {marker} [{sp.id}] {sp.name}")
                if idx >= start_idx:
                    print(f"         max_iterations:  {sp.max_iterations}")
                    print(f"         num_envs:        {sp.num_envs}")
                    print(f"         env.terrain_type: {sp.env.get('terrain_type', 'plane')}")
                    cmd_ranges = sp.env.get("commands", {}).get("ranges", {})
                    if cmd_ranges:
                        print(f"         cmd_ranges:      {cmd_ranges}")
                    print(f"         ppo.learning_rate: {sp.ppo.get('learning_rate', 'N/A')}")
                    print(f"         ppo.entropy_coef:  {sp.ppo.get('entropy_coef', 'N/A')}")
                    print(f"         monitor:         threshold={sp.monitor.get('action_rate_threshold')}, "
                          f"min_iter={sp.monitor.get('min_iterations')}")
                    print(f"         rollback:        threshold={sp.rollback.get('reward_threshold')}, "
                          f"max_retries={sp.rollback.get('max_retries')}")
                print()
            print()

        total = len(self._phase_mgr.all_sub_phases)
        print(f"  Total sub-phases: {total}")
        print(f"  Starting from:    {self._start_id} (index {start_idx})")
        print()
        print("Dry run — no training will be launched.")


# ── Video helpers ──────────────────────────────────────────────────── #


def _find_and_rename_video(run_dir_path: Path, target: Path) -> None:
    """Find the most recent video in a run directory and copy to target."""
    videos_dir = run_dir_path / "videos"
    if not videos_dir.exists():
        return

    mp4_files = sorted(videos_dir.rglob("*.mp4"), key=lambda x: x.stat().st_mtime, reverse=True)
    if mp4_files:
        shutil.copy2(str(mp4_files[0]), str(target))


# ── CLI ────────────────────────────────────────────────────────────── #


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase-Based Training Orchestrator — 5-phase automated RL pipeline",
    )
    parser.add_argument("--plan", type=str, required=True, help="Path to the YAML training plan file")
    parser.add_argument("--project-root", type=str, default=".", help="Root directory of magiclab_rl_lab")
    parser.add_argument("--log-root", type=str, default="logs/rsl_rl/magiclab_z1_12dof_velocity",
                        help="Relative path to the training log root")
    parser.add_argument("--start-from", type=str, default=None, help="Sub-phase ID to start from")
    parser.add_argument("--fresh", action="store_true", help="Ignore saved state and start fresh")
    parser.add_argument("--dry-run", action="store_true", help="Print plan and commands without executing")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run all sub-phases with minimal iterations to verify the full pipeline")
    parser.add_argument("--poll-interval", type=int, default=120, help="Seconds between monitoring polls")
    parser.add_argument("--state-file", type=str, default="orchestrator_state.json",
                        help="Filename for crash-recovery state")
    parser.add_argument("--num-gpus", type=int, default=4, help="Number of GPUs for distributed training")
    return parser.parse_args()


def main():
    args = parse_args()
    orchestrator = PhaseOrchestrator(
        plan_path=args.plan,
        project_root=args.project_root,
        state_path=args.state_file,
        log_root=args.log_root,
        poll_interval=args.poll_interval,
        start_from=args.start_from,
        fresh=args.fresh,
        dry_run=args.dry_run,
        smoke_test=args.smoke_test,
        num_gpus=args.num_gpus,
    )
    orchestrator.run()


if __name__ == "__main__":
    main()
