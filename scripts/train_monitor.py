#!/usr/bin/env python3
"""
Training Overfitting Monitor for MagicBot Z1 RL Training.

Runs alongside the training process on RTX6000, polls checkpoints and
TensorBoard logs, detects overfitting / collapse signals, tracks the
best model, and generates export + transfer commands.

Pure Python — only needs torch + tensorboard + stdlib (already in the
training environment).  Does NOT launch Isaac Sim.

Usage:
    # Realtime: focus on currently training run, compact live updates
    python scripts/train_monitor.py --realtime --terrain gentle

    # One-shot retrospective analysis of all runs
    python scripts/train_monitor.py --once --terrain gentle

    # One-shot analysis of a single finished run
    python scripts/train_monitor.py --once \
        --run_dir logs/rsl_rl/.../2026-05-01_04-50-05_z1_locomotion_s4_gentle_terrain

    # Continuous background monitoring of all runs
    nohup python scripts/train_monitor.py \
        --log_root logs/rsl_rl/magiclab_z1_12dof_velocity \
        --terrain gentle --poll_interval 120 \
        > monitor.log 2>&1 &

    # Continuous with auto-export of best model
    python scripts/train_monitor.py --auto_export --terrain gentle
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# --------------------------------------------------------------------------- #
# 1. Configuration                                                             #
# --------------------------------------------------------------------------- #

TERRAIN_THRESHOLDS = {
    "flat": {
        "expected_best_range": (15000, 30000),
        "reward_ceiling": 49.0,
        "action_rate_warning": -0.8,
    },
    "gentle": {
        "expected_best_range": (20000, 35000),
        "reward_ceiling": 47.0,
        "action_rate_warning": -1.0,
    },
    "rough": {
        "expected_best_range": (25000, 40000),
        "reward_ceiling": 38.0,
        "action_rate_warning": -1.5,
    },
}


@dataclass
class MonitorConfig:
    log_root: str = "logs/rsl_rl/magiclab_z1_12dof_velocity"
    run_dir: Optional[str] = None  # --once mode: analyse a single run
    poll_interval_sec: int = 120
    terrain_type: str = "gentle"

    # Detection thresholds
    reward_decline_pct: float = 20.0
    action_rate_threshold: float = -1.0
    std_min_threshold: float = 0.01
    value_loss_max: float = 100.0
    entropy_collapse_pct: float = 95.0
    min_iterations: int = 1000  # skip detection below this

    # Auto-export
    auto_export: bool = False
    export_script: str = "scripts/export_jit.py"

    # Transfer config
    spark_host: str = "zentek@59.66.25.192"
    rtx_host: str = "phh@192.168.120.155"
    local_tmp: str = r"D:\Desktop_Files\GPU-Train\tmp"

    # Mode
    once: bool = False
    realtime: bool = False

    def __post_init__(self):
        # Override thresholds from terrain presets
        if self.terrain_type in TERRAIN_THRESHOLDS:
            preset = TERRAIN_THRESHOLDS[self.terrain_type]
            if self.action_rate_threshold == -1.0:
                self.action_rate_threshold = preset["action_rate_warning"]


# --------------------------------------------------------------------------- #
# 2. TensorBoard Parser                                                        #
# --------------------------------------------------------------------------- #

# Tags used in rsl-rl logging
TAG_REWARD = "Train/mean_reward"
TAG_ACTION_RATE = "Episode_Reward/action_rate"
TAG_VALUE_LOSS = "Loss/value_function"
TAG_ENTROPY = "Loss/entropy"

# Behavioral metrics (cross-run comparable)
TAG_EP_LEN = "Train/mean_episode_length"
TAG_TIME_OUT = "Episode_Termination/time_out"
TAG_BAD_ORI = "Episode_Termination/bad_orientation"
TAG_VEL_ERR = "Metrics/base_velocity/error_vel_xy"

ALL_TAGS = [TAG_REWARD, TAG_ACTION_RATE, TAG_VALUE_LOSS, TAG_ENTROPY,
            TAG_EP_LEN, TAG_TIME_OUT, TAG_BAD_ORI, TAG_VEL_ERR]


class TensorBoardParser:
    """Read scalar data from TensorBoard event files using EventAccumulator."""

    @staticmethod
    def find_event_file(run_dir: str) -> Optional[str]:
        """Return the directory containing event files (for EventAccumulator)."""
        # rsl-rl writes events directly into the run directory
        p = Path(run_dir)
        if p.is_dir():
            events = list(p.glob("events.out.tfevents.*"))
            if events:
                return str(p)
        return None

    @staticmethod
    def read_metrics(run_dir: str, last_step: int = 0) -> dict[str, list[tuple[int, float]]]:
        """Read all tracked metrics from a run directory.

        Args:
            run_dir: Path to the training run directory.
            last_step: Only read events with step > last_step (incremental).

        Returns:
            Dict mapping tag -> list of (step, value) sorted by step.
        """
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )

        ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
        ea.Reload()

        result: dict[str, list[tuple[int, float]]] = {}
        for tag in ALL_TAGS:
            if tag not in ea.Tags().get("scalars", []):
                result[tag] = []
                continue
            events = ea.Scalars(tag)
            result[tag] = [
                (e.step, e.value) for e in events if e.step > last_step
            ]
        return result


# --------------------------------------------------------------------------- #
# 3. Checkpoint Analyzer                                                       #
# --------------------------------------------------------------------------- #


class CheckpointAnalyzer:
    """Load model_N.pt files and extract training metadata."""

    @staticmethod
    def find_checkpoints(run_dir: str) -> list[Path]:
        """Return sorted list of model_*.pt paths in run_dir."""
        p = Path(run_dir)
        ckpts = sorted(
            p.glob("model_*.pt"),
            key=lambda x: int(x.stem.split("_")[1]),
        )
        return ckpts

    @staticmethod
    def get_iteration(ckpt_path: Path) -> int:
        """Extract iteration number from checkpoint filename."""
        return int(ckpt_path.stem.split("_")[1])

    @staticmethod
    def extract_std(ckpt_path: Path) -> float:
        """Load checkpoint and return mean of the std vector.

        Returns -1.0 if std cannot be found (graceful degradation).
        """
        import torch

        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        state = None
        if "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "actor_state_dict" in ckpt:
            state = ckpt["actor_state_dict"]
        if state is None:
            return -1.0
        std_key = None
        for k in state:
            if k == "std" or k.endswith(".std"):
                std_key = k
                break
        if std_key is None:
            return -1.0
        import torch as _torch

        return float(_torch.tensor(state[std_key]).mean().item())

    @staticmethod
    def extract_weight_stats(ckpt_path: Path) -> dict:
        """Quick weight statistics for diagnostics."""
        import numpy as np
        import torch

        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        state = None
        if "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "actor_state_dict" in ckpt:
            state = ckpt["actor_state_dict"]
        if state is None:
            return {}
        actor_weights = {
            k: v for k, v in state.items()
            if "weight" in k and "std" not in k
        }
        if not actor_weights:
            return {}
        all_abs = torch.cat([v.flatten().abs() for v in actor_weights.values()])
        return {
            "weight_mean_abs": float(all_abs.mean()),
            "weight_max": float(all_abs.max()),
            "weight_std": float(all_abs.std()),
        }


# --------------------------------------------------------------------------- #
# 4. Run State Container                                                       #
# --------------------------------------------------------------------------- #


@dataclass
class RunState:
    """Accumulated state for a single training run."""
    run_name: str
    run_dir: str
    # Metrics from TensorBoard
    rewards: list[tuple[int, float]] = field(default_factory=list)
    action_rates: list[tuple[int, float]] = field(default_factory=list)
    value_losses: list[tuple[int, float]] = field(default_factory=list)
    entropies: list[tuple[int, float]] = field(default_factory=list)
    # Behavioral metrics (cross-run comparable)
    episode_lengths: list[tuple[int, float]] = field(default_factory=list)
    time_outs: list[tuple[int, float]] = field(default_factory=list)
    bad_orientations: list[tuple[int, float]] = field(default_factory=list)
    vel_errors: list[tuple[int, float]] = field(default_factory=list)
    # Checkpoint data
    checkpoints: list[Path] = field(default_factory=list)
    std_values: dict[int, float] = field(default_factory=dict)  # iter -> std
    # Tracking
    peak_reward: float = -float("inf")
    peak_reward_iter: int = 0
    peak_entropy: float = -float("inf")
    best_model_iter: int = 0
    best_model_reward: float = -float("inf")
    # Status
    alerts: list[str] = field(default_factory=list)
    last_checked_step: int = 0
    last_checked_ckpt_iter: int = 0
    overfitting_detected: bool = False
    overfitting_reason: Optional[str] = None


# --------------------------------------------------------------------------- #
# 5. Overfitting Detector                                                      #
# --------------------------------------------------------------------------- #


class OverfittingDetector:
    """Check 5 independent conditions; any triggers an alert."""

    def __init__(self, cfg: MonitorConfig):
        self.cfg = cfg

    def check(self, state: RunState) -> Optional[str]:
        """Return alert reason string, or None if healthy."""
        # Need minimum data
        if not state.rewards:
            return None
        latest_iter = state.rewards[-1][0]
        if latest_iter < self.cfg.min_iterations:
            return None

        # 1. Reward decline from peak
        reason = self._check_reward_decline(state)
        if reason:
            return reason

        # 2. Action rate deterioration
        reason = self._check_action_rate(state)
        if reason:
            return reason

        # 3. Policy std collapse
        reason = self._check_std_collapse(state)
        if reason:
            return reason

        # 4. Value loss explosion
        reason = self._check_value_loss(state)
        if reason:
            return reason

        # 5. Entropy collapse
        reason = self._check_entropy(state)
        if reason:
            return reason

        return None

    def _check_reward_decline(self, state: RunState) -> Optional[str]:
        if state.peak_reward <= 0:
            return None
        # Use median of last 10 entries to filter transient spikes
        recent = [v for _, v in state.rewards[-10:]]
        if not recent:
            return None
        recent_sorted = sorted(recent)
        median_reward = recent_sorted[len(recent_sorted) // 2]
        decline_pct = (state.peak_reward - median_reward) / abs(state.peak_reward) * 100
        if decline_pct > self.cfg.reward_decline_pct:
            return (
                f"Reward declined {decline_pct:.1f}% from peak "
                f"({median_reward:.2f} vs peak {state.peak_reward:.2f}@{state.peak_reward_iter})"
            )
        return None

    def _check_action_rate(self, state: RunState) -> Optional[str]:
        if not state.action_rates:
            return None
        # Use median of last 10 entries to filter transient spikes
        recent = [v for _, v in state.action_rates[-10:]]
        if not recent:
            return None
        recent_sorted = sorted(recent)
        median_ar = recent_sorted[len(recent_sorted) // 2]
        # action_rate penalty is always negative; more negative = worse jitter.
        # Healthy: -0.2 ~ -0.4.  Collapsed: -1.32, -4.75, -21.74.
        if median_ar < self.cfg.action_rate_threshold:
            return (
                f"action_rate = {median_ar:.3f} "
                f"(threshold: {self.cfg.action_rate_threshold})"
            )
        return None

    def _check_std_collapse(self, state: RunState) -> Optional[str]:
        if not state.std_values:
            return None
        latest_std = list(state.std_values.values())[-1]
        if latest_std >= 0 and latest_std < self.cfg.std_min_threshold:
            return f"Policy std = {latest_std:.6f} (threshold: {self.cfg.std_min_threshold})"
        return None

    def _check_value_loss(self, state: RunState) -> Optional[str]:
        if not state.value_losses:
            return None
        # Use median of last 5 entries to filter transient spikes
        recent = [v for _, v in state.value_losses[-5:]]
        if not recent:
            return None
        recent_sorted = sorted(recent)
        median_vl = recent_sorted[len(recent_sorted) // 2]
        if median_vl > self.cfg.value_loss_max:
            return f"Value loss = {median_vl:.2f} (threshold: {self.cfg.value_loss_max})"
        return None

    def _check_entropy(self, state: RunState) -> Optional[str]:
        if not state.entropies or state.peak_entropy <= 0:
            return None
        # Entropy naturally declines during healthy training (from ~24 to ~5).
        # Only flag when entropy drops to near-zero AND decline is extreme.
        recent = [v for _, v in state.entropies[-5:]]
        if not recent:
            return None
        recent_sorted = sorted(recent)
        median_ent = recent_sorted[len(recent_sorted) // 2]
        # Require BOTH: >95% decline from peak AND absolute value < 0.5
        # This avoids false positives on healthy converged policies
        decline_pct = (state.peak_entropy - median_ent) / state.peak_entropy * 100
        if decline_pct > self.cfg.entropy_collapse_pct and median_ent < 0.5:
            return (
                f"Entropy collapsed {decline_pct:.1f}% from peak "
                f"({median_ent:.4f} vs peak {state.peak_entropy:.4f})"
            )
        return None


# --------------------------------------------------------------------------- #
# 6. Best Model Tracker                                                        #
# --------------------------------------------------------------------------- #


class BestModelTracker:
    """Track the checkpoint with the highest smoothed reward."""

    def __init__(self, window: int = 10):
        self.window = window

    def update(self, state: RunState) -> None:
        """Update best model based on smoothed reward curve."""
        if len(state.rewards) < self.window:
            # Not enough data for smoothing — just use raw values
            for step, reward in state.rewards:
                if reward > state.best_model_reward:
                    state.best_model_reward = reward
                    state.best_model_iter = step
            return

        # Compute rolling average
        rewards = state.rewards
        n = len(rewards)
        smoothed: list[tuple[int, float]] = []
        for i in range(n):
            start = max(0, i - self.window + 1)
            window_vals = [v for _, v in rewards[start : i + 1]]
            avg = sum(window_vals) / len(window_vals)
            smoothed.append((rewards[i][0], avg))

        # Find peak of smoothed curve
        for step, sr in smoothed:
            if sr > state.best_model_reward:
                state.best_model_reward = sr
                state.best_model_iter = step


# --------------------------------------------------------------------------- #
# 7. Report Generator                                                          #
# --------------------------------------------------------------------------- #


class ReportGenerator:
    """Format terminal output, file reports, and transfer commands."""

    def __init__(self, cfg: MonitorConfig):
        self.cfg = cfg

    def print_status(self, state: RunState) -> None:
        """Single-line status output."""
        now = datetime.now().strftime("%H:%M")
        latest_reward = state.rewards[-1][1] if state.rewards else 0
        latest_iter = state.rewards[-1][0] if state.rewards else 0
        latest_ar = state.action_rates[-1][1] if state.action_rates else 0
        latest_vl = state.value_losses[-1][1] if state.value_losses else 0
        latest_ent = state.entropies[-1][1] if state.entropies else 0

        # Latest std
        std_str = "N/A"
        if state.std_values:
            std_str = f"{list(state.std_values.values())[-1]:.4f}"

        # Peak info
        peak_str = f"{state.peak_reward:.2f}@{state.peak_reward_iter}"

        status = "OVERFITTING" if state.overfitting_detected else "HEALTHY"

        line = (
            f"[MONITOR {now}] {state.run_name} "
            f"| iter: {latest_iter} "
            f"| reward: {latest_reward:.2f} (peak: {peak_str}) "
            f"| action_rate: {latest_ar:.3f} "
            f"| std: {std_str} "
            f"| vloss: {latest_vl:.4f} "
            f"| entropy: {latest_ent:.4f} "
            f"| {status}"
        )
        print(line)

        # Behavioral metrics line
        if state.time_outs or state.episode_lengths:
            to_val = state.time_outs[-1][1] if state.time_outs else 0
            ep_val = state.episode_lengths[-1][1] if state.episode_lengths else 0
            bo_val = state.bad_orientations[-1][1] if state.bad_orientations else 0
            ve_val = state.vel_errors[-1][1] if state.vel_errors else 0
            print(
                f"  [BEHAVIOR] time_out: {to_val:.1%} | ep_len: {ep_val:.0f}/1000 | "
                f"bad_ori: {bo_val:.1%} | vel_err: {ve_val:.2f} m/s"
            )

    def print_alert(self, state: RunState) -> None:
        """Multi-line alert output."""
        best_ckpt = f"model_{state.best_model_iter}.pt"
        lines = [
            "",
            "=" * 70,
            f"!!! OVERFITTING DETECTED: {state.run_name} !!!",
            f"  Reason: {state.overfitting_reason}",
            f"  Best model: {best_ckpt} (smoothed reward: {state.best_model_reward:.2f})",
            "",
        ]
        lines.extend(self._export_commands(state))
        lines.append("")
        lines.extend(self._transfer_commands(state))
        lines.append("")
        lines.extend(self._spark_play_commands(state))
        lines.append("=" * 70)
        print("\n".join(lines))

    def write_report(self, state: RunState) -> None:
        """Write monitor/report.txt in the run directory."""
        monitor_dir = Path(state.run_dir) / "monitor"
        monitor_dir.mkdir(exist_ok=True)

        report = {
            "run_name": state.run_name,
            "run_dir": state.run_dir,
            "updated_at": datetime.now().isoformat(),
            "latest_iteration": state.rewards[-1][0] if state.rewards else 0,
            "peak_reward": state.peak_reward,
            "peak_reward_iter": state.peak_reward_iter,
            "best_model_iter": state.best_model_iter,
            "best_model_reward": state.best_model_reward,
            "overfitting_detected": state.overfitting_detected,
            "overfitting_reason": state.overfitting_reason,
            "latest_metrics": {
                "reward": state.rewards[-1][1] if state.rewards else None,
                "action_rate": state.action_rates[-1][1] if state.action_rates else None,
                "value_loss": state.value_losses[-1][1] if state.value_losses else None,
                "entropy": state.entropies[-1][1] if state.entropies else None,
                "episode_length": state.episode_lengths[-1][1] if state.episode_lengths else None,
                "time_out": state.time_outs[-1][1] if state.time_outs else None,
                "bad_orientation": state.bad_orientations[-1][1] if state.bad_orientations else None,
                "vel_error": state.vel_errors[-1][1] if state.vel_errors else None,
            },
            "std_values": {str(k): v for k, v in sorted(state.std_values.items())},
        }

        report_path = monitor_dir / "report.txt"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

    def write_overfitting_marker(self, state: RunState) -> None:
        """Create OVERFITTING_DETECTED marker file."""
        monitor_dir = Path(state.run_dir) / "monitor"
        monitor_dir.mkdir(exist_ok=True)

        best_ckpt = f"model_{state.best_model_iter}.pt"
        checkpoint_abs = str(Path(state.run_dir) / best_ckpt)

        marker = {
            "detected_at": datetime.now().isoformat(),
            "reason": state.overfitting_reason,
            "best_model": best_ckpt,
            "best_iteration": state.best_model_iter,
            "best_reward": state.best_model_reward,
            "peak_reward": state.peak_reward,
            "peak_reward_iter": state.peak_reward_iter,
            "current_reward": state.rewards[-1][1] if state.rewards else 0,
            "scp_commands": self._scp_rtx_to_local(state),
            "spark_play_command": self._spark_play_cmd(state),
        }

        marker_path = monitor_dir / "OVERFITTING_DETECTED"
        with open(marker_path, "w") as f:
            json.dump(marker, f, indent=2)

        print(f"[MONITOR] Marker written: {marker_path}")

    # -- Command generators -------------------------------------------------- #

    def _export_commands(self, state: RunState) -> list[str]:
        best_ckpt = Path(state.run_dir) / f"model_{state.best_model_iter}.pt"
        return [
            "  Export:",
            f"    python {self.cfg.export_script} --checkpoint {best_ckpt} --onnx",
        ]

    def _scp_rtx_to_local(self, state: RunState) -> str:
        remote_path = (
            f"{self.cfg.rtx_host}:"
            f"{Path(state.run_dir).resolve() / 'exported' / 'policy.pt'}"
        )
        return f"scp {remote_path} {self.cfg.local_tmp}"

    def _transfer_commands(self, state: RunState) -> list[str]:
        rtx_remote = (
            f"{self.cfg.rtx_host}:"
            f"{Path(state.run_dir).resolve() / 'exported' / 'policy.pt'}"
        )
        local_path = str(Path(self.cfg.local_tmp) / "policy.pt")
        spark_remote = f"{self.cfg.spark_host}:~/RL_Training/Z1_Monitor/policy.pt"
        return [
            "  Transfer:",
            f"    scp {rtx_remote} {local_path}",
            f"    scp {local_path} {spark_remote}",
        ]

    def _spark_play_cmd(self, state: RunState) -> str:
        return (
            f"python scripts/spark_play.py "
            f"--task Magiclab-Z1-12dof-Velocity "
            f"--policy ~/RL_Training/Z1_Monitor/policy.pt "
            f"--video --num_envs 1"
        )

    def _spark_play_commands(self, state: RunState) -> list[str]:
        return [
            "  Spark play:",
            "    " + self._spark_play_cmd(state),
            "",
            "  Spark record (headless):",
            "    export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1",
            "    python scripts/spark_play.py "
            "--task Magiclab-Z1-12dof-Velocity "
            f"--policy ~/RL_Training/Z1_Monitor/policy.pt "
            "--headless --video --video_length 200 --num_envs 1 "
            "--diag_interval 50",
            "",
            "  Retrieve video:",
            f"    scp {self.cfg.spark_host}:~/magiclab_rl_lab/logs/.../videos/play/*.mp4 "
            f"{str(Path(self.cfg.local_tmp) / '..' / 'videos')}",
        ]


# --------------------------------------------------------------------------- #
# 8. Single-run analysis (shared by both modes)                                #
# --------------------------------------------------------------------------- #


def analyze_run(run_dir: str, cfg: MonitorConfig, reporter: ReportGenerator) -> RunState:
    """Fully analyse a single run directory (checkpoints + TensorBoard)."""
    run_name = Path(run_dir).name
    state = RunState(run_name=run_name, run_dir=run_dir)

    # -- Parse TensorBoard events ------------------------------------------ #
    event_dir = TensorBoardParser.find_event_file(run_dir)
    if event_dir:
        try:
            metrics = TensorBoardParser.read_metrics(event_dir)
            state.rewards = metrics.get(TAG_REWARD, [])
            state.action_rates = metrics.get(TAG_ACTION_RATE, [])
            state.value_losses = metrics.get(TAG_VALUE_LOSS, [])
            state.entropies = metrics.get(TAG_ENTROPY, [])
            state.episode_lengths = metrics.get(TAG_EP_LEN, [])
            state.time_outs = metrics.get(TAG_TIME_OUT, [])
            state.bad_orientations = metrics.get(TAG_BAD_ORI, [])
            state.vel_errors = metrics.get(TAG_VEL_ERR, [])
        except Exception as e:
            print(f"[WARN] Failed to parse TensorBoard events in {run_dir}: {e}")
    else:
        print(f"[WARN] No TensorBoard events found in {run_dir}")

    # -- Analyse checkpoints ----------------------------------------------- #
    ckpts = CheckpointAnalyzer.find_checkpoints(run_dir)
    state.checkpoints = ckpts

    # Extract std from every checkpoint (may be slow for many ckpts)
    for ckpt_path in ckpts:
        iteration = CheckpointAnalyzer.get_iteration(ckpt_path)
        try:
            std_val = CheckpointAnalyzer.extract_std(ckpt_path)
            state.std_values[iteration] = std_val
        except Exception as e:
            print(f"[WARN] Failed to extract std from {ckpt_path.name}: {e}")

    # -- Update tracking --------------------------------------------------- #
    # Peak reward
    for step, reward in state.rewards:
        if reward > state.peak_reward:
            state.peak_reward = reward
            state.peak_reward_iter = step

    # Peak entropy
    for _, ent in state.entropies:
        if ent > state.peak_entropy:
            state.peak_entropy = ent

    # Best model (smoothed)
    tracker = BestModelTracker(window=10)
    tracker.update(state)

    # -- Overfitting detection --------------------------------------------- #
    detector = OverfittingDetector(cfg)
    reason = detector.check(state)
    if reason:
        state.overfitting_detected = True
        state.overfitting_reason = reason

    # -- Reports ----------------------------------------------------------- #
    reporter.print_status(state)
    if state.overfitting_detected:
        reporter.print_alert(state)
        reporter.write_overfitting_marker(state)
    reporter.write_report(state)

    return state


# --------------------------------------------------------------------------- #
# 9. Continuous monitoring loop                                                #
# --------------------------------------------------------------------------- #


def find_run_dirs(log_root: str) -> list[str]:
    """Find all training run directories under log_root."""
    root = Path(log_root)
    if not root.is_dir():
        return []
    # rsl-rl run dirs match: YYYY-MM-DD_HH-MM-SS_*
    runs = sorted(
        [str(d) for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")]
    )
    return runs


def find_active_run(log_root: str) -> Optional[str]:
    """Find the currently active training run.

    Strategy: find the run directory whose latest checkpoint or TensorBoard
    event file has the most recent modification time.  Only considers runs
    whose latest checkpoint was modified within the last 30 minutes.

    Returns the run directory path, or None if no active run found.
    """
    root = Path(log_root)
    if not root.is_dir():
        return None

    best_run: Optional[str] = None
    best_mtime: float = 0.0
    cutoff = time.time() - 1800  # 30 minutes ago

    for run_dir in root.iterdir():
        if not run_dir.is_dir() or run_dir.name.startswith("."):
            continue
        # Check modification time of the most recent checkpoint or event file
        latest_mtime = 0.0
        for pattern in ["model_*.pt", "events.out.tfevents.*"]:
            for f in run_dir.glob(pattern):
                mtime = f.stat().st_mtime
                if mtime > latest_mtime:
                    latest_mtime = mtime
        if latest_mtime > cutoff and latest_mtime > best_mtime:
            best_mtime = latest_mtime
            best_run = str(run_dir)

    return best_run


def incremental_update(state: RunState, cfg: MonitorConfig) -> None:
    """Incrementally read new data since last check."""
    # -- New TensorBoard data ---------------------------------------------- #
    event_dir = TensorBoardParser.find_event_file(state.run_dir)
    if event_dir:
        try:
            metrics = TensorBoardParser.read_metrics(
                event_dir, last_step=state.last_checked_step
            )
            # Append new data
            state.rewards.extend(metrics.get(TAG_REWARD, []))
            state.action_rates.extend(metrics.get(TAG_ACTION_RATE, []))
            state.value_losses.extend(metrics.get(TAG_VALUE_LOSS, []))
            state.entropies.extend(metrics.get(TAG_ENTROPY, []))
            state.episode_lengths.extend(metrics.get(TAG_EP_LEN, []))
            state.time_outs.extend(metrics.get(TAG_TIME_OUT, []))
            state.bad_orientations.extend(metrics.get(TAG_BAD_ORI, []))
            state.vel_errors.extend(metrics.get(TAG_VEL_ERR, []))
        except Exception as e:
            print(f"[WARN] TensorBoard parse error for {state.run_name}: {e}")

    # Update last_checked_step
    if state.rewards:
        state.last_checked_step = state.rewards[-1][0]

    # -- New checkpoints --------------------------------------------------- #
    all_ckpts = CheckpointAnalyzer.find_checkpoints(state.run_dir)
    state.checkpoints = all_ckpts

    # Only analyse checkpoints we haven't seen yet
    for ckpt_path in all_ckpts:
        iteration = CheckpointAnalyzer.get_iteration(ckpt_path)
        if iteration > state.last_checked_ckpt_iter and iteration not in state.std_values:
            try:
                std_val = CheckpointAnalyzer.extract_std(ckpt_path)
                state.std_values[iteration] = std_val
            except Exception:
                pass

    if all_ckpts:
        state.last_checked_ckpt_iter = max(
            CheckpointAnalyzer.get_iteration(c) for c in all_ckpts
        )

    # -- Update tracking --------------------------------------------------- #
    for step, reward in state.rewards:
        if reward > state.peak_reward:
            state.peak_reward = reward
            state.peak_reward_iter = step

    for _, ent in state.entropies:
        if ent > state.peak_entropy:
            state.peak_entropy = ent

    tracker = BestModelTracker(window=10)
    tracker.update(state)


def run_realtime(cfg: MonitorConfig) -> None:
    """Realtime monitoring: focus on the single active training run.

    Auto-detects which run is currently training (most recent checkpoint
    modified within 30 minutes), then polls it every 30 seconds showing
    compact live-updating status.  Exits automatically when training stops
    (no new data for 3 consecutive polls).
    """
    poll_interval = min(cfg.poll_interval_sec, 30)  # cap at 30s for realtime
    stale_threshold = 3  # exit after this many polls with no new data

    # Auto-detect active run
    active_dir = cfg.run_dir or find_active_run(cfg.log_root)
    if not active_dir:
        print("[REALTIME] No active training run detected.")
        print("[REALTIME] Start training first, or use --run_dir to specify.")
        sys.exit(1)

    run_name = Path(active_dir).name
    print(f"[REALTIME] Monitoring: {run_name}")
    print(f"[REALTIME] Poll every {poll_interval}s, auto-exit after {stale_threshold} stale polls")
    print(f"[REALTIME] Ctrl+C to stop")
    print()

    # Initialize state
    state = RunState(run_name=run_name, run_dir=active_dir)
    reporter = ReportGenerator(cfg)
    detector = OverfittingDetector(cfg)
    tracker = BestModelTracker(window=10)

    # Initial full load
    incremental_update(state, cfg)
    prev_iter = state.last_checked_step
    prev_time = time.time()

    reporter.print_status(state)
    print()

    stale_count = 0

    while True:
        try:
            time.sleep(poll_interval)

            # Incremental read
            incremental_update(state, cfg)

            new_iter = state.last_checked_step
            now = time.time()
            elapsed = now - prev_time

            # Detect new data
            if new_iter > prev_iter:
                stale_count = 0
                iters_delta = new_iter - prev_iter
                speed = iters_delta / elapsed if elapsed > 0 else 0
                prev_iter = new_iter
                prev_time = now
            else:
                stale_count += 1
                speed = 0
                iters_delta = 0

            # Overfitting check
            if not state.overfitting_detected:
                reason = detector.check(state)
                if reason:
                    state.overfitting_detected = True
                    state.overfitting_reason = reason
                    reporter.print_alert(state)
                    reporter.write_overfitting_marker(state)
                    if cfg.auto_export:
                        _auto_export_best(state, cfg)

            # Compact realtime status
            ts = datetime.now().strftime("%H:%M:%S")
            latest_reward = state.rewards[-1][1] if state.rewards else 0
            latest_ar = state.action_rates[-1][1] if state.action_rates else 0
            latest_ent = state.entropies[-1][1] if state.entropies else 0
            status = "OVERFITTING" if state.overfitting_detected else "HEALTHY"

            # Trend arrow
            if len(state.rewards) >= 2:
                recent = [v for _, v in state.rewards[-5:]]
                older = [v for _, v in state.rewards[-10:-5]] if len(state.rewards) >= 10 else recent
                recent_avg = sum(recent) / len(recent)
                older_avg = sum(older) / len(older)
                if recent_avg > older_avg * 1.02:
                    trend = "↑"
                elif recent_avg < older_avg * 0.98:
                    trend = "↓"
                else:
                    trend = "→"
            else:
                trend = "?"

            speed_str = f"{speed:.0f} iter/s" if speed > 0 else "stale"

            # Behavioral metrics for realtime
            to_str = f"to:{state.time_outs[-1][1]:.0%}" if state.time_outs else ""
            ep_str = f"ep:{state.episode_lengths[-1][1]:.0f}" if state.episode_lengths else ""
            ve_str = f"ve:{state.vel_errors[-1][1]:.2f}" if state.vel_errors else ""
            extra = " | ".join(filter(None, [to_str, ep_str, ve_str]))

            print(
                f"[{ts}] iter {new_iter:>6} | "
                f"reward {latest_reward:>7.2f} {trend} | "
                f"peak {state.peak_reward:.2f}@{state.peak_reward_iter} | "
                f"best {state.best_model_reward:.2f}@{state.best_model_iter} | "
                f"ar {latest_ar:.3f} | ent {latest_ent:.1f} | "
                f"{speed_str} | {status}"
            )
            if extra:
                print(f"  [BEHAVIOR] {extra}")

            # Update report file
            reporter.write_report(state)

            # Auto-exit if training stopped
            if stale_count >= stale_threshold:
                print()
                print(f"[REALTIME] No new data for {stale_threshold * poll_interval}s — training appears stopped.")
                print(f"[REALTIME] Final: iter {new_iter}, reward {latest_reward:.2f}, best model_{state.best_model_iter}.pt ({state.best_model_reward:.2f})")
                break

        except KeyboardInterrupt:
            print(f"\n[REALTIME] Stopped. Latest: iter {state.last_checked_step}")
            break
        except Exception as e:
            print(f"[REALTIME] Error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(poll_interval)


def run_continuous(cfg: MonitorConfig) -> None:
    """Main continuous monitoring loop."""
    reporter = ReportGenerator(cfg)
    detector = OverfittingDetector(cfg)

    # Active runs being monitored
    active_runs: dict[str, RunState] = {}

    print(f"[MONITOR] Starting continuous monitoring")
    print(f"[MONITOR] Log root: {cfg.log_root}")
    print(f"[MONITOR] Terrain: {cfg.terrain_type}")
    print(f"[MONITOR] Poll interval: {cfg.poll_interval_sec}s")
    print(f"[MONITOR] Auto-export: {cfg.auto_export}")
    print()

    while True:
        try:
            # Discover all run directories
            run_dirs = find_run_dirs(cfg.log_root)

            for run_dir in run_dirs:
                run_name = Path(run_dir).name

                # Skip runs that already triggered overfitting (already handled)
                marker = Path(run_dir) / "monitor" / "OVERFITTING_DETECTED"
                if marker.exists() and run_name not in active_runs:
                    continue  # already detected, skip

                if run_name not in active_runs:
                    # New run — do initial full analysis
                    print(f"[MONITOR] New run detected: {run_name}")
                    state = RunState(run_name=run_name, run_dir=run_dir)
                    active_runs[run_name] = state
                    # Incremental first load (reads all data since step 0)
                    incremental_update(state, cfg)
                else:
                    state = active_runs[run_name]
                    incremental_update(state, cfg)

                # Check for overfitting
                if not state.overfitting_detected:
                    reason = detector.check(state)
                    if reason:
                        state.overfitting_detected = True
                        state.overfitting_reason = reason
                        reporter.print_alert(state)
                        reporter.write_overfitting_marker(state)

                        # Auto-export best model
                        if cfg.auto_export:
                            _auto_export_best(state, cfg)

                # Print status
                reporter.print_status(state)
                reporter.write_report(state)

            print()  # blank line between cycles
            time.sleep(cfg.poll_interval_sec)

        except KeyboardInterrupt:
            print("\n[MONITOR] Stopped by user.")
            break
        except Exception as e:
            print(f"[MONITOR] Error in monitoring loop: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(cfg.poll_interval_sec)


def _auto_export_best(state: RunState, cfg: MonitorConfig) -> None:
    """Call export_jit.py for the best model checkpoint."""
    best_ckpt = Path(state.run_dir) / f"model_{state.best_model_iter}.pt"
    if not best_ckpt.exists():
        print(f"[WARN] Best model checkpoint not found: {best_ckpt}")
        return

    cmd = [
        sys.executable,
        cfg.export_script,
        "--checkpoint",
        str(best_ckpt),
    ]
    print(f"[MONITOR] Auto-exporting best model: {best_ckpt.name}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(f"[MONITOR] Export successful")
        else:
            print(f"[MONITOR] Export failed: {result.stderr}")
    except Exception as e:
        print(f"[MONITOR] Export error: {e}")


# --------------------------------------------------------------------------- #
# 10. CLI entry point                                                          #
# --------------------------------------------------------------------------- #


def parse_args():
    parser = argparse.ArgumentParser(
        description="Training Overfitting Monitor for MagicBot Z1 RL Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Mode
    parser.add_argument(
        "--once", action="store_true",
        help="One-shot analysis mode (analyse existing run, then exit)",
    )
    parser.add_argument(
        "--realtime", action="store_true",
        help="Realtime mode: auto-detect active run, compact live updates every 30s",
    )

    # Paths
    parser.add_argument(
        "--log_root", type=str,
        default="logs/rsl_rl/magiclab_z1_12dof_velocity",
        help="Root directory containing training runs (for continuous mode)",
    )
    parser.add_argument(
        "--run_dir", type=str, default=None,
        help="Single run directory (for --once mode)",
    )

    # Terrain
    parser.add_argument(
        "--terrain", type=str, default="gentle",
        choices=["flat", "gentle", "rough"],
        help="Terrain type (adjusts thresholds)",
    )

    # Polling
    parser.add_argument(
        "--poll_interval", type=int, default=120,
        help="Polling interval in seconds (default: 120)",
    )

    # Thresholds (override terrain presets)
    parser.add_argument("--reward_decline_pct", type=float, default=20.0)
    parser.add_argument("--action_rate_threshold", type=float, default=-1.0)
    parser.add_argument("--std_min_threshold", type=float, default=0.01)
    parser.add_argument("--value_loss_max", type=float, default=100.0)
    parser.add_argument("--entropy_collapse_pct", type=float, default=80.0)
    parser.add_argument("--min_iterations", type=int, default=1000)

    # Auto-export
    parser.add_argument(
        "--auto_export", action="store_true",
        help="Auto-export best model when overfitting detected",
    )
    parser.add_argument(
        "--export_script", type=str, default="scripts/export_jit.py",
        help="Path to export_jit.py script",
    )

    # Transfer
    parser.add_argument("--spark_host", type=str, default="zentek@59.66.25.192")
    parser.add_argument("--rtx_host", type=str, default="phh@192.168.120.155")
    parser.add_argument(
        "--local_tmp", type=str,
        default=r"D:\Desktop_Files\GPU-Train\tmp",
        help="Local Windows staging directory for SCP transfers",
    )

    return parser.parse_args()


def _write_best_models_json(results: list[RunState], cfg: MonitorConfig) -> None:
    """Write a consolidated best_models.json at log_root level.

    This file is read by the gpu-train skill's --sim command to auto-find
    the best checkpoint for any training version.
    """
    log_root = Path(cfg.log_root)
    if not log_root.is_dir():
        log_root = Path(cfg.run_dir).parent if cfg.run_dir else None
        if not log_root:
            return

    entries = []
    for state in results:
        # Skip runs with no meaningful data
        if not state.rewards or state.peak_reward == -float("inf"):
            continue
        # Extract short version name from run directory
        # e.g. "2026-05-01_04-50-05_z1_locomotion_s4_gentle_terrain" -> "s4_gentle"
        run_dir_name = Path(state.run_dir).name
        version = run_dir_name
        # Try to extract version identifier
        for part in run_dir_name.split("_"):
            if part.startswith("v") and any(c.isdigit() for c in part):
                # Collect remaining parts after version number
                idx = run_dir_name.index(part)
                version = run_dir_name[idx:]
                # Remove z1_locomotion_ prefix
                version = version.replace("z1_locomotion_", "")
                break

        best_ckpt = f"model_{state.best_model_iter}.pt"
        checkpoint_path = str(Path(state.run_dir) / best_ckpt)

        entries.append({
            "version": version,
            "run_dir": run_dir_name,
            "status": "HEALTHY" if not state.overfitting_detected else "OVERFITTING",
            "overfitting_reason": state.overfitting_reason,
            "latest_iteration": state.rewards[-1][0] if state.rewards else 0,
            "latest_reward": state.rewards[-1][1] if state.rewards else 0,
            "peak_reward": round(state.peak_reward, 2),
            "peak_reward_iter": state.peak_reward_iter,
            "best_model_iteration": state.best_model_iter,
            "best_model_reward": round(state.best_model_reward, 2),
            "best_model_file": best_ckpt,
            "checkpoint_path": checkpoint_path,
            "latest_action_rate": state.action_rates[-1][1] if state.action_rates else None,
            "latest_std": list(state.std_values.values())[-1] if state.std_values else None,
            "latest_time_out": state.time_outs[-1][1] if state.time_outs else None,
            "latest_episode_length": state.episode_lengths[-1][1] if state.episode_lengths else None,
            "latest_bad_orientation": state.bad_orientations[-1][1] if state.bad_orientations else None,
            "latest_vel_error": state.vel_errors[-1][1] if state.vel_errors else None,
        })

    # Sort by best reward descending
    entries.sort(key=lambda x: x["best_model_reward"], reverse=True)

    output = {
        "generated_at": datetime.now().isoformat(),
        "log_root": str(log_root),
        "total_runs": len(results),
        "models": entries,
    }

    output_path = log_root / "best_models.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"[MONITOR] Best models summary written to: {output_path}")
    print(f"[MONITOR] {len(entries)} runs with data, top: ", end="")
    if entries:
        top = entries[0]
        print(f"{top['version']} -> {top['best_model_file']} (reward: {top['best_model_reward']})")
    else:
        print("(no data)")


def main():
    args = parse_args()

    cfg = MonitorConfig(
        log_root=args.log_root,
        run_dir=args.run_dir,
        poll_interval_sec=args.poll_interval,
        terrain_type=args.terrain,
        reward_decline_pct=args.reward_decline_pct,
        action_rate_threshold=args.action_rate_threshold,
        std_min_threshold=args.std_min_threshold,
        value_loss_max=args.value_loss_max,
        entropy_collapse_pct=args.entropy_collapse_pct,
        min_iterations=args.min_iterations,
        auto_export=args.auto_export,
        export_script=args.export_script,
        spark_host=args.spark_host,
        rtx_host=args.rtx_host,
        local_tmp=args.local_tmp,
        once=args.once,
        realtime=args.realtime,
    )

    reporter = ReportGenerator(cfg)

    if cfg.realtime:
        # -- Realtime mode ----------------------------------------------- #
        run_realtime(cfg)
    elif cfg.once:
        # -- One-shot mode ------------------------------------------------ #
        if cfg.run_dir:
            run_dirs = [cfg.run_dir]
        else:
            run_dirs = find_run_dirs(cfg.log_root)

        if not run_dirs:
            print(f"[ERROR] No run directories found")
            sys.exit(1)

        print(f"[MONITOR] One-shot analysis of {len(run_dirs)} run(s)")
        print()

        results: list[RunState] = []
        for rd in run_dirs:
            try:
                state = analyze_run(rd, cfg, reporter)
                results.append(state)
                print()
            except Exception as e:
                print(f"[ERROR] Failed to analyse {rd}: {e}")
                import traceback
                traceback.print_exc()

        # Write consolidated best_models.json at log_root level
        _write_best_models_json(results, cfg)
    else:
        # -- Continuous mode ---------------------------------------------- #
        run_continuous(cfg)


if __name__ == "__main__":
    main()
