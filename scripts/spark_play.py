#!/usr/bin/env python3
"""
Universal play script for Spark (DGX Spark, aarch64) — JIT policy inference with IsaacLab.

Loads a JIT-exported policy and runs it in IsaacLab environment for evaluation/video recording.
Does NOT use OnPolicyRunner, avoiding rsl-rl version compatibility issues.

This script is designed to run on Spark (IsaacLab 0.54.3 + IsaacSim 5.1.0) with JIT models
exported from RTX6000 (rsl-rl 3.0.1) or any other platform.

IMPORTANT: By default, actions are NOT clipped (matching Isaac Lab training behavior where
RslRlVecEnvWrapper uses clip_actions=(-100, 100)). Use --clip_actions to add clipping if needed.
Do NOT use clip=1.0 for accurate Isaac Lab evaluation — that creates a mismatch with training.

Usage:
    # Interactive (VNC) — no clipping (matches training):
    python scripts/spark_play.py \
        --task Magiclab-Z1-12dof-Velocity \
        --policy logs/rsl_rl/<RUN>/exported/policy.pt \
        --real-time --num_envs 4

    # Headless video recording:
    python scripts/spark_play.py \
        --task Magiclab-Z1-12dof-Velocity \
        --policy logs/rsl_rl/<RUN>/exported/policy.pt \
        --headless --video --video_length 200 --num_envs 1

    # With action clipping (for comparison with MuJoCo clip=1.0):
    python scripts/spark_play.py \
        --task Magiclab-Z1-12dof-Velocity \
        --policy logs/rsl_rl/<RUN>/exported/policy.pt \
        --clip_actions 1.0 --num_envs 1

    # Diagnostic mode (print action/velocity stats every 50 steps):
    python scripts/spark_play.py \
        --task Magiclab-Z1-12dof-Velocity \
        --policy logs/rsl_rl/<RUN>/exported/policy.pt \
        --diag_interval 50 --num_envs 1

Note: On Spark, set LD_PRELOAD before running:
    export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1
"""

import argparse
import os
import sys
import time
import traceback

from isaaclab.app import AppLauncher

# --- Parse arguments before launching Isaac Sim ---
parser = argparse.ArgumentParser(description="Universal Spark play script with JIT policy (no rsl-rl dependency).")
parser.add_argument("--task", type=str, required=True, help="IsaacLab task name (e.g. Magiclab-Z1-12dof-Velocity)")
parser.add_argument("--policy", type=str, required=True, help="Path to JIT-exported policy (.pt)")
parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments")
parser.add_argument("--video", action="store_true", default=False, help="Record video")
parser.add_argument("--video_length", type=int, default=200, help="Video length in steps")
parser.add_argument("--max_steps", type=int, default=1000, help="Max steps when not recording video")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time")
parser.add_argument("--seed", type=int, default=None, help="Random seed")
parser.add_argument("--clip_actions", type=float, default=None,
                    help="Clip actions to [-N, N]. Default: None (no clip, matches Isaac Lab training). "
                         "Use 1.0 to compare with MuJoCo clip behavior.")
parser.add_argument("--diag_interval", type=int, default=50,
                    help="Print diagnostic stats every N steps (raw action magnitudes, velocity, etc.)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Enable cameras for video recording
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# === Imports after Isaac Sim launch ===
import torch
import numpy as np
import gymnasium as gym

import isaaclab_tasks  # noqa: F401
import magiclab_rl_lab.tasks  # noqa: F401 — registers task environments

from magiclab_rl_lab.utils.parser_cfg import parse_env_cfg


def main():
    try:
        task_name = args_cli.task
        policy_path = args_cli.policy

        print(f"[INFO] Task: {task_name}", flush=True)
        print(f"[INFO] Policy: {policy_path}", flush=True)

        # --- Load JIT policy ---
        device = args_cli.device
        policy = torch.jit.load(policy_path, map_location=device)
        policy.eval()
        print(f"[INFO] JIT policy loaded on {device}", flush=True)

        # --- Create environment ---
        entry_point = "play_env_cfg_entry_point"
        # Fallback to default env_cfg_entry_point if play variant not available
        try:
            env_cfg = parse_env_cfg(
                task_name,
                device=device,
                num_envs=args_cli.num_envs,
                entry_point_key=entry_point,
            )
        except Exception:
            env_cfg = parse_env_cfg(
                task_name,
                device=device,
                num_envs=args_cli.num_envs,
            )

        if args_cli.seed is not None:
            env_cfg.seed = args_cli.seed

        log_dir = os.path.dirname(os.path.dirname(policy_path))  # exported/ -> run_dir
        if hasattr(env_cfg, 'log_dir'):
            env_cfg.log_dir = log_dir

        print("[INFO] Creating environment...", flush=True)
        if args_cli.video:
            env = gym.make(task_name, cfg=env_cfg, render_mode="rgb_array")
        else:
            env = gym.make(task_name, cfg=env_cfg)
        print(f"[INFO] Obs space: {env.observation_space}, Action space: {env.action_space}", flush=True)

        # --- Reset environment and get initial obs ---
        obs, info = env.reset()
        print(f"[INFO] Environment reset. Obs shape: {obs.shape if hasattr(obs, 'shape') else type(obs)}", flush=True)

        # --- Camera tracking setup ---
        cam_ctx = None
        if args_cli.video:
            try:
                from isaaclab.sim import SimulationContext
                cam_ctx = SimulationContext.instance()
                robot_pos = env.unwrapped.scene["robot"].data.root_pos_w[0].cpu().numpy()
                cam_ctx.set_camera_view(
                    eye=[robot_pos[0] + 1.0, robot_pos[1] + 3.5, robot_pos[2] + 1.5],
                    target=[robot_pos[0] + 0.5, robot_pos[1], robot_pos[2] + 0.5],
                )
                print(f"[INFO] Camera tracking enabled. Robot at ({robot_pos[0]:.1f}, {robot_pos[1]:.1f}, {robot_pos[2]:.1f})", flush=True)
            except Exception as e:
                print(f"[WARN] Camera tracking setup failed: {e}. Video may not show robot.", flush=True)
                cam_ctx = None

        # --- Renderer warmup: skip initial black frames ---
        if args_cli.video:
            warmup_steps = 20
            print(f"[INFO] Renderer warmup: running {warmup_steps} steps to prime the viewport...", flush=True)
            with torch.inference_mode():
                for _ in range(warmup_steps):
                    policy_obs = obs["policy"] if isinstance(obs, dict) else obs
                    actions = policy(policy_obs)
                    if cam_ctx is not None:
                        try:
                            robot_pos = env.unwrapped.scene["robot"].data.root_pos_w[0].cpu().numpy()
                            cam_ctx.set_camera_view(
                                eye=[robot_pos[0] + 1.0, robot_pos[1] + 3.5, robot_pos[2] + 1.5],
                                target=[robot_pos[0] + 0.5, robot_pos[1], robot_pos[2] + 0.5],
                            )
                        except Exception:
                            pass
                    obs, _, _, _, _ = env.step(actions)
            print(f"[INFO] Warmup done. Resetting environment for recording...", flush=True)
            obs, info = env.reset()

        # --- Wrap for video recording (after warmup, so no black frames) ---
        if args_cli.video:
            video_folder = os.path.join(log_dir, "videos", "play")
            video_kwargs = {
                "video_folder": video_folder,
                "step_trigger": lambda step: step == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            print(f"[INFO] Recording video to: {video_folder}", flush=True)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        # Determine obs format — rsl-rl wrapper returns dict obs["policy"]
        # but raw gym env may return tensor directly
        def get_policy_obs(observation):
            """Extract the policy observation from the environment output."""
            if isinstance(observation, dict):
                return observation["policy"]
            return observation

        # --- Inference loop ---
        max_steps = args_cli.video_length if args_cli.video else args_cli.max_steps
        clip_val = args_cli.clip_actions
        timestep = 0
        start_time = time.time()
        diag_interval = args_cli.diag_interval

        # Action statistics accumulators for final summary
        all_raw_actions = []

        clip_str = f"{clip_val}" if clip_val is not None else "None (no clip, matches training)"
        print(f"[INFO] Running rollout ({max_steps} steps, clip_actions={clip_str})...", flush=True)

        while simulation_app.is_running():
            with torch.inference_mode():
                policy_obs = get_policy_obs(obs)
                raw_actions = policy(policy_obs)

                # Collect raw action stats (env 0 only)
                if diag_interval > 0:
                    all_raw_actions.append(raw_actions[0].cpu().clone())

                # Clip actions only if explicitly requested
                if clip_val is not None and clip_val > 0:
                    actions = torch.clamp(raw_actions, -clip_val, clip_val)
                else:
                    actions = raw_actions

                # Update camera to follow robot (env 0) during video recording
                if cam_ctx is not None:
                    try:
                        robot_pos = env.unwrapped.scene["robot"].data.root_pos_w[0].cpu().numpy()
                        cam_ctx.set_camera_view(
                            eye=[robot_pos[0] + 1.0, robot_pos[1] + 3.5, robot_pos[2] + 1.5],
                            target=[robot_pos[0] + 0.5, robot_pos[1], robot_pos[2] + 0.5],
                        )
                    except Exception:
                        pass

                # Step environment
                obs, reward, terminated, truncated, info = env.step(actions)

            timestep += 1

            # Diagnostic output
            if diag_interval > 0 and timestep % diag_interval == 0:
                raw_act = raw_actions[0].cpu().numpy()
                act = actions[0].cpu().numpy()
                print(
                    f"[DIAG] Step {timestep:5d}/{max_steps} | "
                    f"raw_action: mean_abs={np.abs(raw_act).mean():.2f}, "
                    f"range=[{raw_act.min():.1f}, {raw_act.max():.1f}] | "
                    f"clipped_action: mean_abs={np.abs(act).mean():.2f}, "
                    f"range=[{act.min():.2f}, {act.max():.2f}]",
                    flush=True,
                )

            if timestep >= max_steps:
                break

            # Real-time pacing
            if args_cli.real_time:
                step_dt = env.unwrapped.step_dt
                elapsed = time.time() - start_time
                expected = timestep * step_dt
                if expected - elapsed > 0:
                    time.sleep(expected - elapsed)

        elapsed = time.time() - start_time
        print(f"[INFO] Rollout done. Steps: {timestep}, Time: {elapsed:.1f}s", flush=True)
        if elapsed > 0:
            print(f"[INFO] FPS: {timestep / elapsed:.0f}", flush=True)

        # Final action statistics summary
        if all_raw_actions:
            all_raw = torch.stack(all_raw_actions).numpy()
            print(f"\n[SUMMARY] Raw action statistics across rollout:", flush=True)
            print(f"  Overall mean_abs: {np.abs(all_raw).mean():.2f}", flush=True)
            print(f"  Per-step mean_abs range: [{np.abs(all_raw).mean(axis=1).min():.2f}, "
                  f"{np.abs(all_raw).mean(axis=1).max():.2f}]", flush=True)
            print(f"  Global min: {all_raw.min():.2f}, max: {all_raw.max():.2f}", flush=True)
            print(f"  Percentiles — p50: {np.median(np.abs(all_raw)):.2f}, "
                  f"p90: {np.percentile(np.abs(all_raw), 90):.2f}, "
                  f"p99: {np.percentile(np.abs(all_raw), 99):.2f}", flush=True)

        env.close()
        print("[INFO] Environment closed.", flush=True)

    except Exception as e:
        print(f"[ERROR] {e}", flush=True)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
    simulation_app.close()
