# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""Custom play script for MagicBot Z1 12DOF locomotion with video recording (bypasses Hydra).

Supports two inference modes:
  --checkpoint model.pt  →  OnPolicyRunner (rsl-rl, existing behavior)
  --policy policy.pt     →  JIT inference (from spark_play.py, no rsl-rl dependency)

Camera tracking and renderer warmup are ported from spark_play.py (only active with --video).
"""

import argparse
import os
import sys
import time
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play MagicBot Z1 12DOF locomotion RL agent with video recording.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (OnPolicyRunner mode).")
parser.add_argument("--policy", type=str, default=None, help="Path to JIT-exported policy (.pt).")
parser.add_argument("--seed", type=int, default=None, help="Random seed.")
parser.add_argument("--video", action="store_true", default=False, help="Record video.")
parser.add_argument("--video_length", type=int, default=1000, help="Length of recorded video (in steps).")
parser.add_argument("--max_steps", type=int, default=800, help="Max steps when not recording video.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable Fabric and fall back to USD I/O.",
)
parser.add_argument("--no_camera_track", action="store_true", default=False,
                    help="Disable camera tracking (camera stays at default position).")
parser.add_argument("--camera_distance", type=float, default=3.5,
                    help="Camera distance from robot for tracking (default: 3.5).")
parser.add_argument("--camera_height", type=float, default=1.5,
                    help="Camera height above robot for tracking (default: 1.5).")
parser.add_argument(
    "--terrain",
    type=str,
    default="default",
    choices=["default", "plane", "stair", "stair_ring"],
    help=(
        "Terrain override for recording. "
        "'default' uses the env config; "
        "'plane' forces flat plane; "
        "'stair' uses official pyramid stairs with one-way upward traversal; "
        "'stair_ring' keeps the older ring-style inverted stairs."
    ),
)
parser.add_argument(
    "--command_resample_time",
    type=float,
    default=None,
    help="Override base velocity command resampling time in seconds.",
)
parser.add_argument("--fixed_vel_x", type=float, default=None, help="Fix commanded forward velocity (m/s).")
parser.add_argument("--fixed_vel_y", type=float, default=None, help="Fix commanded lateral velocity (m/s).")
parser.add_argument("--fixed_vel_yaw", type=float, default=None, help="Fix commanded yaw rate (rad/s).")
parser.add_argument(
    "--stair_start_x_min",
    type=float,
    default=-3.4,
    help="For stair terrain, minimum robot start x so it walks before climbing.",
)
parser.add_argument(
    "--stair_start_x_max",
    type=float,
    default=-1.8,
    help="For stair terrain, maximum robot start x so it walks before climbing.",
)
parser.add_argument(
    "--active_gpu",
    type=int,
    default=None,
    help="Explicit Omniverse renderer GPU index. Defaults to the index parsed from --device.",
)
parser.add_argument(
    "--physics_gpu",
    type=int,
    default=None,
    help="Explicit PhysX GPU index. Defaults to the index parsed from --device.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Validate: exactly one of --checkpoint or --policy must be provided
if not args_cli.checkpoint and not args_cli.policy:
    parser.error("Must provide --checkpoint or --policy.")
if args_cli.checkpoint and args_cli.policy:
    parser.error("--checkpoint and --policy are mutually exclusive.")

# Enable cameras for video recording
if args_cli.video:
    args_cli.enable_cameras = True


def _parse_gpu_index_from_device(device: str | None) -> int | None:
    if not device or not device.startswith("cuda:"):
        return None
    try:
        return int(device.split(":", 1)[1])
    except ValueError:
        return None


gpu_index = _parse_gpu_index_from_device(getattr(args_cli, "device", None))
if args_cli.active_gpu is None:
    args_cli.active_gpu = gpu_index
if args_cli.physics_gpu is None:
    args_cli.physics_gpu = gpu_index
args_cli.multi_gpu = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# === Imports after Isaac Sim launch ===
import torch
import numpy as np
import gymnasium as gym
import importlib
import importlib.metadata as metadata
import isaaclab.terrains as terrain_gen

import isaaclab_tasks  # noqa: F401
import magiclab_rl_lab.tasks  # noqa: F401 - registers Magiclab-Z1-12dof-Velocity

# RobotPlayEnvCfg lives under robots/z1/12dof/ (digit-starting dir, can't use import statement)
# Resolve dynamically through gym registration
TASK_NAME = "Magiclab-Z1-12dof-Velocity"
_spec = gym.spec(TASK_NAME)
_entry = _spec.kwargs.get("play_env_cfg_entry_point", _spec.kwargs["env_cfg_entry_point"])
_mod_path, _cls_name = _entry.rsplit(":", 1)
RobotPlayEnvCfg = getattr(importlib.import_module(_mod_path), _cls_name)
from magiclab_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg

# JIT mode flag
USE_JIT = args_cli.policy is not None

if not USE_JIT:
    from rsl_rl.runners import OnPolicyRunner
    try:
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    except ImportError:
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        handle_deprecated_rsl_rl_cfg = None

    CKPT = args_cli.checkpoint
    print(f"[INFO] Mode: OnPolicyRunner (checkpoint)", flush=True)
    print(f"[INFO] Loading checkpoint: {CKPT}", flush=True)
else:
    CKPT = args_cli.policy
    print(f"[INFO] Mode: JIT policy (no rsl-rl dependency)", flush=True)
    print(f"[INFO] Loading policy: {CKPT}", flush=True)


def _update_camera(cam_ctx, env, cam_dist, cam_height):
    """Update camera to follow env-0 robot."""
    robot_pos = env.unwrapped.scene["robot"].data.root_pos_w[0].cpu().numpy()
    cam_ctx.set_camera_view(
        eye=[robot_pos[0] + 1.0, robot_pos[1] + cam_dist, robot_pos[2] + cam_height],
        target=[robot_pos[0] + 0.5, robot_pos[1], robot_pos[2] + 0.5],
    )


def _extract_command_row(command):
    """Convert a command tensor/array into env-0 velocity floats."""
    if command is None:
        return None, None, None

    try:
        if hasattr(command, "detach"):
            command = command.detach()
        if hasattr(command, "cpu"):
            command = command.cpu()
        command = np.asarray(command)
        row = command[0] if command.ndim > 1 else command
        if row.shape[0] < 3:
            return None, None, None
        return float(row[0]), float(row[1]), float(row[2])
    except Exception:
        return None, None, None


def _get_vel_commands(env):
    """Read current base velocity command from env-0."""
    unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
    command_manager = getattr(unwrapped, "command_manager", None)
    if command_manager is None:
        return None, None, None

    try:
        return _extract_command_row(command_manager.get_command("base_velocity"))
    except Exception:
        return _extract_command_row(getattr(command_manager, "command", None))


def _save_vel_log(vel_log, log_dir):
    """Persist per-frame velocity commands next to the recorded video."""
    import json

    if not vel_log or not log_dir:
        return

    video_folder = os.path.join(log_dir, "videos", "play")
    os.makedirs(video_folder, exist_ok=True)
    data = {
        "type": "isaac_lab",
        "fps": 50,
        "num_frames": len(vel_log),
        "vel_x": [cmd[0] for cmd in vel_log],
        "vel_y": [cmd[1] for cmd in vel_log],
        "vel_yaw": [cmd[2] for cmd in vel_log],
    }
    for name in ("sweep.json", "vel_commands.json"):
        out_path = os.path.join(video_folder, name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    print(f"[INFO] Velocity log saved: {os.path.join(video_folder, 'sweep.json')}", flush=True)


def _maybe_flush_vel_log(vel_log, log_dir, step):
    """Flush velocity log periodically so data survives forced shutdowns."""
    if vel_log is None or step <= 0:
        return
    if step % 50 == 0:
        _save_vel_log(vel_log, log_dir)


def _set_equal_command_range(command_cfg, vel_x: float, vel_y: float, vel_yaw: float):
    ranges_cls = type(command_cfg.ranges)
    fixed_ranges = ranges_cls(
        lin_vel_x=(vel_x, vel_x),
        lin_vel_y=(vel_y, vel_y),
        ang_vel_z=(vel_yaw, vel_yaw),
    )
    command_cfg.ranges = fixed_ranges
    if hasattr(command_cfg, "limit_ranges"):
        command_cfg.limit_ranges = fixed_ranges
    command_cfg.rel_standing_envs = 0.0
    command_cfg.rel_heading_envs = 0.0
    command_cfg.heading_command = False


def _build_stair_terrain_cfg(terrain_gen):
    return terrain_gen.TerrainGeneratorCfg(
        size=(8.0, 8.0),
        border_width=20.0,
        num_rows=1,
        num_cols=1,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        difficulty_range=(1.0, 1.0),
        use_cache=False,
        sub_terrains={
            "stair": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                proportion=1.0,
                step_height_range=(0.08, 0.12),
                step_width=0.35,
                platform_width=2.0,
                border_width=1.0,
                holes=False,
            ),
        },
    )


def _build_one_way_stair_terrain_cfg(terrain_gen):
    return terrain_gen.TerrainGeneratorCfg(
        size=(8.0, 8.0),
        border_width=20.0,
        num_rows=1,
        num_cols=1,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        difficulty_range=(1.0, 1.0),
        use_cache=False,
        sub_terrains={
            "stair": terrain_gen.MeshPyramidStairsTerrainCfg(
                proportion=1.0,
                step_height_range=(0.08, 0.12),
                step_width=0.35,
                platform_width=2.0,
                border_width=1.0,
                holes=False,
            ),
        },
    )


def _apply_recording_reset_policy(env_cfg):
    reset_base = getattr(getattr(env_cfg, "events", None), "reset_base", None)
    if reset_base is not None and hasattr(reset_base, "params"):
        pose_range = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        if args_cli.terrain == "stair":
            # Keep heading aligned with the +x climb direction, but randomize the
            # start distance so the robot walks before reaching the stairs.
            x_min = min(args_cli.stair_start_x_min, args_cli.stair_start_x_max)
            x_max = max(args_cli.stair_start_x_min, args_cli.stair_start_x_max)
            pose_range["x"] = (x_min, x_max)
            pose_range["y"] = (0.0, 0.0)
        reset_base.params["pose_range"] = pose_range
        reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        print(
            "[INFO] Reset override:"
            f" pose={reset_base.params['pose_range']},"
            f" velocity={reset_base.params['velocity_range']}",
            flush=True,
        )

    push_robot = getattr(getattr(env_cfg, "events", None), "push_robot", None)
    if push_robot is not None:
        if hasattr(push_robot, "interval_range_s"):
            push_robot.interval_range_s = (1.0e9, 1.0e9)
        if hasattr(push_robot, "params"):
            push_robot.params["velocity_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0)}
        print("[INFO] Disabled push_robot disturbance for recording.", flush=True)


def _apply_recording_overrides(env_cfg):
    if args_cli.terrain == "plane":
        env_cfg.scene.terrain.terrain_type = "plane"
        env_cfg.scene.terrain.terrain_generator = None
        if hasattr(env_cfg.scene.terrain, "max_init_terrain_level"):
            env_cfg.scene.terrain.max_init_terrain_level = 0
        print("[INFO] Terrain override: plane", flush=True)
    elif args_cli.terrain == "stair":
        env_cfg.scene.terrain.terrain_type = "generator"
        env_cfg.scene.terrain.terrain_generator = _build_one_way_stair_terrain_cfg(terrain_gen)
        if hasattr(env_cfg.scene.terrain, "max_init_terrain_level"):
            env_cfg.scene.terrain.max_init_terrain_level = 0
        print("[INFO] Terrain override: stair (official pyramid stairs, one-way +x climb)", flush=True)
    elif args_cli.terrain == "stair_ring":
        env_cfg.scene.terrain.terrain_type = "generator"
        env_cfg.scene.terrain.terrain_generator = _build_stair_terrain_cfg(terrain_gen)
        if hasattr(env_cfg.scene.terrain, "max_init_terrain_level"):
            env_cfg.scene.terrain.max_init_terrain_level = 0
        print("[INFO] Terrain override: stair_ring (inverted ring stairs)", flush=True)
    else:
        print("[INFO] Terrain override: default", flush=True)

    if args_cli.command_resample_time is not None:
        env_cfg.commands.base_velocity.resampling_time_range = (
            args_cli.command_resample_time,
            args_cli.command_resample_time,
        )
        print(
            f"[INFO] Command resample time override: {args_cli.command_resample_time:.3f}s",
            flush=True,
        )

    fixed_velocity_requested = any(
        value is not None for value in (args_cli.fixed_vel_x, args_cli.fixed_vel_y, args_cli.fixed_vel_yaw)
    )
    if fixed_velocity_requested:
        vel_x = 0.0 if args_cli.fixed_vel_x is None else args_cli.fixed_vel_x
        vel_y = 0.0 if args_cli.fixed_vel_y is None else args_cli.fixed_vel_y
        vel_yaw = 0.0 if args_cli.fixed_vel_yaw is None else args_cli.fixed_vel_yaw
        _set_equal_command_range(env_cfg.commands.base_velocity, vel_x, vel_y, vel_yaw)
        print(
            f"[INFO] Fixed velocity override: vx={vel_x:.3f}, vy={vel_y:.3f}, yaw={vel_yaw:.3f}",
            flush=True,
        )
    else:
        ranges = env_cfg.commands.base_velocity.ranges
        print(
            "[INFO] Command range:"
            f" vx={ranges.lin_vel_x}, vy={ranges.lin_vel_y}, yaw={ranges.ang_vel_z}",
            flush=True,
        )

    _apply_recording_reset_policy(env_cfg)


def main():
    try:
        # === Env config (play mode) ===
        env_cfg = RobotPlayEnvCfg()

        # --- Fix headless rendering: post-process USD to clear instancing ---
        # URDF converter's ImportConfig has no set_make_instanceable() (unlike MJCF).
        # Instead, we monkey-patch UrdfConverter.__init__ to modify the generated USD
        # file AFTER conversion but BEFORE the spawner loads it into the stage.
        import isaaclab.sim.converters.urdf_converter as _uc
        _orig_uc_init = _uc.UrdfConverter.__init__
        def _patched_uc_init(self, cfg):
            _orig_uc_init(self, cfg)
            # Post-process: open the generated USD and clear all instanceable attrs
            try:
                from pxr import Usd as _Usd
                usd_path = str(self.usd_path)
                if os.path.exists(usd_path):
                    stage = _Usd.Stage.Open(usd_path)
                    count = 0
                    for prim in stage.Traverse():
                        if prim.IsInstanceable():
                            prim.SetInstanceable(False)
                            count += 1
                    if count > 0:
                        stage.GetRootLayer().Save()
                        print(f"[INFO] USD post-process: cleared instanceable on {count} prims", flush=True)
            except Exception as e:
                print(f"[WARN] USD post-process failed: {e}", flush=True)
        _uc.UrdfConverter.__init__ = _patched_uc_init

        # Clear USD cache to force re-conversion with our patch
        import shutil, glob
        for d in glob.glob("/tmp/IsaacLab/usd_*"):
            shutil.rmtree(d, ignore_errors=True)
        print("[INFO] Cleared USD cache", flush=True)

        env_cfg.scene.robot.spawn.force_usd_conversion = True
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        env_cfg.sim.use_fabric = not args_cli.disable_fabric
        env_cfg.scene.num_envs = args_cli.num_envs
        if args_cli.seed is not None:
            env_cfg.seed = args_cli.seed
        _apply_recording_overrides(env_cfg)

        log_dir = os.path.dirname(CKPT) if not USE_JIT else os.path.dirname(os.path.dirname(CKPT))
        if hasattr(env_cfg, 'log_dir'):
            env_cfg.log_dir = log_dir

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

        # === Camera tracking setup (from spark_play.py) ===
        cam_ctx = None
        cam_dist = args_cli.camera_distance
        cam_height = args_cli.camera_height

        if not args_cli.no_camera_track:
            try:
                from isaaclab.sim import SimulationContext
                cam_ctx = SimulationContext.instance()
                _update_camera(cam_ctx, env, cam_dist, cam_height)
                robot_pos = env.unwrapped.scene["robot"].data.root_pos_w[0].cpu().numpy()
                print(f"[INFO] Camera tracking enabled (dist={cam_dist}, height={cam_height}). "
                      f"Robot at ({robot_pos[0]:.1f}, {robot_pos[1]:.1f}, {robot_pos[2]:.1f})", flush=True)
            except Exception as e:
                print(f"[WARN] Camera tracking setup failed: {e}. Camera will stay at default position.", flush=True)
                cam_ctx = None
        else:
            # Static angled view
            try:
                from isaaclab.sim import SimulationContext
                cam_ctx_static = SimulationContext.instance()
                robot_pos = env.unwrapped.scene["robot"].data.root_pos_w[0].cpu().numpy()
                cam_ctx_static.set_camera_view(
                    eye=[robot_pos[0] + 12.0, robot_pos[1] + 12.0, robot_pos[2] + 15.0],
                    target=[robot_pos[0], robot_pos[1], robot_pos[2]],
                )
                print("[INFO] Camera tracking disabled. Static angled view set (45 deg overhead).", flush=True)
            except Exception as e:
                print(f"[INFO] Camera tracking disabled. Static view setup failed: {e}.", flush=True)

        # === Branch: JIT vs OnPolicyRunner ===
        if USE_JIT:
            _run_jit(env, cam_ctx, cam_dist, cam_height, log_dir)
        else:
            _run_onpolicy(env, cam_ctx, cam_dist, cam_height, log_dir)

    except Exception as e:
        print(f"[ERROR] Exception: {e}", flush=True)
        traceback.print_exc()
        raise


def _run_jit(env, cam_ctx, cam_dist, cam_height, log_dir):
    """JIT policy inference path (ported from spark_play.py)."""
    device = args_cli.device
    policy = torch.jit.load(CKPT, map_location=device)
    policy.eval()
    print(f"[INFO] JIT policy loaded on {device}", flush=True)

    # Reset environment
    obs, info = env.reset()

    # === Renderer warmup (from spark_play.py) ===
    if args_cli.video:
        warmup_steps = 20
        print(f"[INFO] Renderer warmup: running {warmup_steps} steps to prime the viewport...", flush=True)
        with torch.no_grad():
            for _ in range(warmup_steps):
                policy_obs = obs["policy"] if isinstance(obs, dict) else obs
                actions = policy(policy_obs)
                if cam_ctx is not None:
                    try:
                        _update_camera(cam_ctx, env, cam_dist, cam_height)
                    except Exception:
                        pass
                obs, _, _, _, _ = env.step(actions)
        print("[INFO] Warmup done. Resetting environment for recording...", flush=True)
        obs, info = env.reset()

    # === Wrap for video recording (after warmup, so no black frames) ===
    if args_cli.video:
        video_folder = os.path.join(log_dir, "videos", "play")
        video_kwargs = {
            "video_folder": video_folder,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
            "fps": 50,
        }
        print(f"[INFO] Recording video to: {video_folder}", flush=True)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    def get_policy_obs(observation):
        if isinstance(observation, dict):
            return observation["policy"]
        return observation

    # === Inference loop ===
    max_steps = args_cli.video_length if args_cli.video else args_cli.max_steps
    timestep = 0
    start_time = time.time()
    vel_log = [] if args_cli.video else None

    print(f"[INFO] Running rollout ({max_steps} steps, JIT mode)...", flush=True)

    while simulation_app.is_running():
        with torch.no_grad():
            policy_obs = get_policy_obs(obs)
            actions = policy(policy_obs)

            # Update camera to follow robot (env 0)
            if cam_ctx is not None:
                try:
                    _update_camera(cam_ctx, env, cam_dist, cam_height)
                except Exception:
                    pass

            obs, reward, terminated, truncated, info = env.step(actions)

        if vel_log is not None:
            vel_log.append(_get_vel_commands(env))
            _maybe_flush_vel_log(vel_log, log_dir, timestep + 1)

        timestep += 1
        if timestep >= max_steps:
            break

    elapsed = time.time() - start_time
    print(f"[INFO] Rollout done. Steps: {timestep}, Time: {elapsed:.1f}s", flush=True)
    if elapsed > 0:
        print(f"[INFO] FPS: {timestep / elapsed:.0f}", flush=True)

    _save_vel_log(vel_log, log_dir)
    env.close()
    print("[INFO] Environment closed. Video saved.", flush=True)


def _run_onpolicy(env, cam_ctx, cam_dist, cam_height, log_dir):
    """OnPolicyRunner inference path (existing behavior)."""
    agent_cfg = BasePPORunnerCfg()
    if handle_deprecated_rsl_rl_cfg is not None:
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    # === Renderer warmup ===
    if args_cli.video:
        warmup_steps = 20
        print(f"[INFO] Renderer warmup: running {warmup_steps} steps to prime the viewport...", flush=True)
        # We need a temporary runner for warmup before wrapping with RecordVideo
        temp_wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        temp_runner = OnPolicyRunner(
            temp_wrapped, agent_cfg.to_dict(),
            log_dir=None, device=agent_cfg.device,
        )
        temp_runner.load(CKPT)
        temp_policy = temp_runner.get_inference_policy(device=env.unwrapped.device)

        obs = temp_wrapped.get_observations()
        with torch.no_grad():
            for _ in range(warmup_steps):
                actions = temp_policy(obs)
                if cam_ctx is not None:
                    try:
                        _update_camera(cam_ctx, env, cam_dist, cam_height)
                    except Exception:
                        pass
                obs, _, _, _ = temp_wrapped.step(actions)
        print("[INFO] Warmup done.", flush=True)

    # === Wrap for video recording ===
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
            "fps": 50,
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
    max_steps = args_cli.video_length if args_cli.video else args_cli.max_steps
    vel_log = [] if args_cli.video else None

    start_time = time.time()
    print(f"[INFO] Running rollout (max {max_steps} steps, OnPolicyRunner mode)...", flush=True)

    while simulation_app.is_running():
        with torch.no_grad():
            actions = policy(obs)

            # Update camera to follow robot (env 0)
            if cam_ctx is not None:
                try:
                    _update_camera(cam_ctx, env, cam_dist, cam_height)
                except Exception:
                    pass

            obs, rewards, dones, info = env_wrapped.step(actions)

        if vel_log is not None:
            vel_log.append(_get_vel_commands(env))
            _maybe_flush_vel_log(vel_log, log_dir, timestep + 1)

        timestep += 1
        if timestep >= max_steps:
            break

    elapsed = time.time() - start_time
    print(f"[INFO] Rollout done. Steps: {timestep}, Time: {elapsed:.1f}s", flush=True)
    if elapsed > 0:
        print(f"[INFO] FPS: {timestep / elapsed:.0f}", flush=True)

    _save_vel_log(vel_log, log_dir)
    env.close()
    print("[INFO] Environment closed. Video saved.", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
