"""Subprocess manager for ``train.py`` training runs.

Supports both single-GPU (``train.py``) and multi-GPU distributed
(``torchrun train_multigpu.py``) launch modes.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class TrainingLauncher:
    """Launch and manage a training subprocess."""

    def __init__(
        self,
        train_script: str = "scripts/rsl_rl/train.py",
        multigpu_script: str = "scripts/rsl_rl/train_multigpu.py",
        log_dir: str = "logs",
        python_executable: str = "python",
        cwd: Optional[str] = None,
    ):
        self._train_script = train_script
        self._multigpu_script = multigpu_script
        self._log_dir = Path(log_dir)
        self._python = python_executable
        self._cwd = cwd
        self._log_dir.mkdir(parents=True, exist_ok=True)

    # -- Launch -------------------------------------------------------------- #

    def launch(
        self,
        run_name: str,
        max_iterations: int,
        num_envs: int = 4096,
        device: str = "cuda:0",
        task: str = "Magiclab-Z1-12dof-Velocity",
        checkpoint: Optional[str] = None,
        num_gpus: int = 1,
        master_port: int = 29502,
        extra_args: Optional[list[str]] = None,
    ) -> subprocess.Popen:
        """Start training as a subprocess and return the :class:`Popen`.

        If *num_gpus* > 1, uses ``torchrun`` with ``train_multigpu.py``.
        Otherwise uses ``train.py`` on a single GPU.
        """
        log_file = self._log_dir / f"train_{run_name}.log"

        if num_gpus > 1:
            cmd = self._build_multigpu_cmd(
                run_name, max_iterations, num_envs, device, task,
                checkpoint, num_gpus, master_port, extra_args,
            )
        else:
            cmd = self._build_single_cmd(
                run_name, max_iterations, num_envs, device, task,
                checkpoint, extra_args,
            )

        logger.info("Launching training: %s", " ".join(cmd))
        logger.info("Log file: %s", log_file)

        fd = open(log_file, "w", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            stdout=fd,
            stderr=subprocess.STDOUT,
            cwd=self._cwd,
            preexec_fn=os.setsid,
        )
        logger.info("Training PID=%d started (%d GPU(s))", proc.pid, num_gpus)
        return proc

    def _build_single_cmd(self, run_name, max_iterations, num_envs, device,
                          task, checkpoint, extra_args):
        cmd = [
            self._python, "-u",
            self._train_script,
            "--task", task,
            "--headless",
            "--device", device,
            "--num_envs", str(num_envs),
            "--max_iterations", str(max_iterations),
            "--run_name", run_name,
        ]
        if checkpoint:
            cmd += ["--checkpoint", str(checkpoint)]
        if extra_args:
            cmd.extend(extra_args)
        return cmd

    def _build_multigpu_cmd(self, run_name, max_iterations, num_envs, device,
                            task, checkpoint, num_gpus, master_port, extra_args):
        # Extract starting GPU index from device string
        gpu_start = 0
        if device.startswith("cuda:"):
            try:
                gpu_start = int(device.split(":")[1])
            except ValueError:
                pass

        cmd = [
            "torchrun",
            f"--nproc_per_node={num_gpus}",
            f"--master_port={master_port}",
            self._multigpu_script,
            f"--task={task}",
            f"--run_name={run_name}",
            "--headless",
            "--distributed",
            f"--num_envs={num_envs}",
            f"--max_iterations={max_iterations}",
        ]
        if checkpoint:
            # For multigpu, use --resume --load_run=... --checkpoint=model_N.pt
            ckpt_path = Path(checkpoint)
            run_dir = ckpt_path.parent
            ckpt_name = ckpt_path.name
            cmd += [
                "--resume",
                f"--load_run={run_dir.name}",
                f"--checkpoint={ckpt_name}",
            ]
        if extra_args:
            cmd.extend(extra_args)
        return cmd

    # -- Query --------------------------------------------------------------- #

    @staticmethod
    def is_running(pid: Optional[int]) -> bool:
        """Return *True* if *pid* is a live, non-zombie process."""
        if pid is None:
            return False
        try:
            os.kill(pid, 0)
        except (ProcessLookupError, PermissionError, OSError):
            return False
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
            state_char = stat.split(" ")[2]
            if state_char in ("Z",):
                return False
        except (FileNotFoundError, IndexError, PermissionError):
            pass
        return True

    # -- Stop ---------------------------------------------------------------- #

    @staticmethod
    def graceful_stop(pid: int, timeout: int = 60) -> bool:
        """Send SIGTERM, wait *timeout* seconds, then SIGKILL."""
        logger.info("Stopping PID %d (SIGTERM, timeout=%ds)", pid, timeout)

        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            return True

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not TrainingLauncher.is_running(pid):
                logger.info("PID %d exited gracefully", pid)
                return True
            time.sleep(2)

        logger.warning("PID %d did not exit in %ds — sending SIGKILL", pid, timeout)
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            return True

        time.sleep(2)
        return not TrainingLauncher.is_running(pid)
