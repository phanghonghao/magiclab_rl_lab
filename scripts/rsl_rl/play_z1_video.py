# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""Custom play script for MagicBot Z1 12DOF locomotion with video recording (bypasses Hydra)."""

import argparse
import sys
import os
import time
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play MagicBot Z1 12DOF locomotion RL agent with video recording.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Random seed.")
parser.add_argument("--video", action="store_true", default=False, help="Record video.")
parser.add_argument("--video_length", type=int, default=200, help="Length of recorded video (in steps).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Enable cameras for video recording
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# === Imports after Isaac Sim launch ===
import torch
import gymnasium as gym
import importlib.metadata as metadata

import isaaclab_tasks  # noqa: F401
import magiclab_rl_lab.tasks  # noqa: F401 - registers Magiclab-Z1-12dof-Velocity

from magiclab_rl_lab.tasks.locomotion.robots.z1_12dof.velocity_env_cfg import RobotPlayEnvCfg
from magiclab_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

CKPT = args_cli.checkpoint
print(f"[INFO] Loading checkpoint: {CKPT}", flush=True)

TASK_NAME = "Magiclab-Z1-12dof-Velocity"


def main():
    try:
        # === Env config (play mode) ===
        env_cfg = RobotPlayEnvCfg()
        env_cfg.scene.num_envs = args_cli.num_envs
        if args_cli.seed is not None:
            env_cfg.seed = args_cli.seed

        log_dir = os.path.dirname(CKPT)
        env_cfg.log_dir = log_dir

        # === Agent config ===
        agent_cfg = BasePPORunnerCfg()
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

        # === Create environment with render_mode for video ===
        print("[INFO] Creating environment...", flush=True)
        if args_cli.video:
            env = gym.make(
                TASK_NAME,
                cfg=env_cfg,
                render_mode="rgb_array",
            )
        else:
            env = gym.make(TASK_NAME, cfg=env_cfg)
        print(f"[INFO] Obs space: {env.observation_space}, Action space: {env.action_space}", flush=True)

        # === Wrap for video recording ===
        if args_cli.video:
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "play"),
                "step_trigger": lambda step: step == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            print(f"[INFO] Recording video to: {video_kwargs['video_folder']}", flush=True)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        # === Wrap for rsl-rl ===
        env_wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        # === Create runner and load model ===
        runner = OnPolicyRunner(
            env_wrapped, agent_cfg.to_dict(),
            log_dir=None, device=agent_cfg.device,
        )
        runner.load(CKPT)
        print("[INFO] Model loaded successfully.", flush=True)

        # === Get inference policy ===
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        print("[INFO] Inference policy ready.", flush=True)

        # === Run rollout ===
        obs = env_wrapped.get_observations()
        timestep = 0
        max_steps = args_cli.video_length if args_cli.video else 800

        start_time = time.time()
        print(f"[INFO] Running rollout (max {max_steps} steps)...", flush=True)

        while simulation_app.is_running():
            with torch.inference_mode():
                actions = policy(obs)
                obs, rewards, dones, info = env_wrapped.step(actions)

            timestep += 1
            if args_cli.video and timestep >= args_cli.video_length:
                break
            if not args_cli.video and timestep >= max_steps:
                break

        elapsed = time.time() - start_time
        print(f"[INFO] Rollout done. Steps: {timestep}, Time: {elapsed:.1f}s", flush=True)
        if elapsed > 0:
            print(f"[INFO] FPS: {timestep / elapsed:.0f}", flush=True)

        # Video is saved automatically by RecordVideo on env close
        env.close()
        print("[INFO] Environment closed. Video saved.", flush=True)

    except Exception as e:
        print(f"[ERROR] Exception: {e}", flush=True)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
    simulation_app.close()
