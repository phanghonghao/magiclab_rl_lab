#!/usr/bin/env python3
"""Humanoid-Gym style MuJoCo Sim2Sim for MagicBot Z1 (12 DOF).

Inspired by roboterax/humanoid-gym/scripts/sim2sim.py architecture:
  - Class-based config (Z1SimConfig, Z1RobotConfig, Z1ObsConfig)
  - Clean run_mujoco(policy, cfg) entry point
  - Direct PD torque control via data.ctrl

Keeps Isaac Lab per-term observation history layout for model compatibility.

Usage:
    # Record video (headless EGL, no X Server needed):
    python sim2sim/mujoco_sim2sim.py --sim MuJoCo \\
        --mjcf ../magicbot-z1_description/mjcf/MAGICBOTZ1.xml \\
        --checkpoint logs/rsl_rl/<run>/model_49999.pt \\
        --record /tmp/z1_s4_gentle.mp4

    # Live viewer (needs display):
    python sim2sim/mujoco_sim2sim.py --sim MuJoCo \\
        --mjcf ../magicbot-z1_description/mjcf/MAGICBOTZ1.xml \\
        --checkpoint logs/rsl_rl/<run>/model_49999.pt

    # With keyboard velocity control:
    python sim2sim/mujoco_sim2sim.py --sim MuJoCo \\
        --mjcf ../magicbot-z1_description/mjcf/MAGICBOTZ1.xml \\
        --checkpoint logs/rsl_rl/<run>/model_49999.pt \\
        --keyboard
"""

import argparse
import math
import os
import sys
import time
from collections import deque

# MUST set before importing mujoco for EGL offscreen rendering
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np


# ============================================================================
# Config classes (Humanoid-Gym style)
# ============================================================================

class Z1SimConfig:
    """Simulation parameters matching Isaac Lab training."""
    dt = 0.002             # Physics timestep (s)
    decimation = 10        # Control decimation: policy runs at dt*decimation = 50Hz
    control_dt = dt * decimation  # 0.02s
    default_duration = 60.0


class Z1RobotConfig:
    """Robot PD gains, joint layout, and physical parameters."""
    num_actions = 12
    init_height = 0.69

    joint_names = [
        "JOINT_HIP_PITCH_L", "JOINT_HIP_ROLL_L", "JOINT_HIP_YAW_L",
        "JOINT_KNEE_PITCH_L", "JOINT_ANKLE_PITCH_L", "JOINT_ANKLE_ROLL_L",
        "JOINT_HIP_PITCH_R", "JOINT_HIP_ROLL_R", "JOINT_HIP_YAW_R",
        "JOINT_KNEE_PITCH_R", "JOINT_ANKLE_PITCH_R", "JOINT_ANKLE_ROLL_R",
    ]
    actuator_names = [
        "left_hip_pitch_actuator", "left_hip_roll_actuator", "left_hip_yaw_actuator",
        "left_knee_actuator", "left_ankle_pitch_actuator", "left_ankle_roll_actuator",
        "right_hip_pitch_actuator", "right_hip_roll_actuator", "right_hip_yaw_actuator",
        "right_knee_actuator", "right_ankle_pitch_actuator", "right_ankle_roll_actuator",
    ]

    kps = np.array([100, 100, 100, 150, 60, 60,
                     100, 100, 100, 150, 60, 60], dtype=np.float64)
    kds = np.array([4, 4, 4, 5, 3, 3,
                     4, 4, 4, 5, 3, 3], dtype=np.float64)
    tau_limit = np.array([120, 120, 120, 120, 50, 50,
                          120, 120, 120, 120, 50, 50], dtype=np.float64)
    default_joint_pos = np.array([
        -0.35, 0.0, 0.0, 0.7, -0.35, 0.0,   # left leg
        -0.35, 0.0, 0.0, 0.7, -0.35, 0.0,   # right leg
    ], dtype=np.float64)
    armature = np.array([
        0.02863, 0.02863, 0.02863, 0.02863, 0.01503, 0.01503,
        0.02863, 0.02863, 0.02863, 0.02863, 0.01503, 0.01503,
    ])


class Z1ObsConfig:
    """Observation parameters matching Isaac Lab training."""
    # Single frame: ang_vel(3) + gravity(3) + cmd(3) + jpos(12) + jvel(12) + act(12) + gait(2)
    num_single_obs = 47
    frame_stack = 5
    num_observations = num_single_obs * frame_stack  # 235

    # Isaac Lab per-term history layout
    term_dims = [3, 3, 3, 12, 12, 12, 2]  # ang_vel, gravity, cmd, jpos, jvel, act, gait

    ang_vel_scale = 0.2
    joint_vel_scale = 0.05
    action_scale = 0.25
    clip_actions = 1.0
    gait_period = 0.6


# ============================================================================
# Math utilities
# ============================================================================

def quat_to_rot_matrix(quat_wxyz):
    """Convert quaternion (w,x,y,z) to 3x3 rotation matrix."""
    w, x, y, z = quat_wxyz
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z),  2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),       1 - 2*(x*x + y*y)],
    ])


def compute_gait_phase(sim_time, vel_cmd, period=Z1ObsConfig.gait_period):
    """Compute gait phase observation matching Isaac Lab training."""
    if np.linalg.norm(vel_cmd) < 0.02:
        return np.array([1.0, 1.0])  # standing: both feet stance
    phase = (sim_time % period) / period
    sin_pos = math.sin(2.0 * math.pi * phase)
    return np.array([1.0 if sin_pos >= 0 else 0.0,
                     1.0 if sin_pos < 0 else 0.0])


# ============================================================================
# Observation
# ============================================================================

def get_obs(data, robot, obs_cfg, last_action, vel_cmd, sim_time):
    """Build a single observation frame (47-dim), matching Isaac Lab training.

    Returns:
        np.ndarray of shape (47,)
    """
    obs = np.zeros(obs_cfg.num_single_obs)

    # Rotation matrix: standard formula returns body-to-world (R_bw)
    # To transform world vectors to body frame, use R_bw.T (world-to-body)
    quat = data.qpos[3:7].copy()  # (w, x, y, z)
    R_bw = quat_to_rot_matrix(quat)  # body-to-world
    R_wb = R_bw.T                    # world-to-body (what we need)

    # 1. Angular velocity in body frame (3), scale=0.2
    omega_world = data.qvel[3:6].copy()
    obs[0:3] = (R_wb @ omega_world) * obs_cfg.ang_vel_scale

    # 2. Projected gravity in body frame (3)
    obs[3:6] = R_wb @ np.array([0.0, 0.0, -1.0])

    # 3. Velocity commands (3)
    obs[6:9] = vel_cmd

    # 4. Joint positions relative to default (12)
    joint_pos = np.array([data.qpos[a] for a in robot.qpos_addr])
    obs[9:21] = joint_pos - robot.default_joint_pos

    # 5. Joint velocities (12), scale=0.05
    joint_vel = np.array([data.qvel[a] for a in robot.dof_addr])
    obs[21:33] = joint_vel * obs_cfg.joint_vel_scale

    # 6. Last action (12)
    obs[33:45] = last_action

    # 7. Gait phase (2)
    obs[45:47] = compute_gait_phase(sim_time, vel_cmd)

    return obs


class ObservationBuffer:
    """Observation history with Isaac Lab per-term interleaving layout.

    Isaac Lab layout: [term0_h0..h4, term1_h0..h4, ..., termN_h0..h4]
    NOT simple frame stacking: [frame0(47), frame1(47), ...]
    """

    def __init__(self, frame_stack, num_single_obs, term_dims):
        self.buffer = np.zeros((frame_stack, num_single_obs))
        self.frame_stack = frame_stack
        self.num_single_obs = num_single_obs
        self.term_dims = term_dims

    def reset(self, initial_obs):
        for i in range(self.frame_stack):
            self.buffer[i] = initial_obs

    def append(self, obs):
        self.buffer = np.roll(self.buffer, -1, axis=0)
        self.buffer[-1] = obs

    def get(self):
        """Return observation in Isaac Lab per-term history layout."""
        total = self.frame_stack * self.num_single_obs
        result = np.empty(total, dtype=np.float64)
        src_col = 0
        dst = 0
        for term_dim in self.term_dims:
            for h in range(self.frame_stack):
                result[dst:dst + term_dim] = self.buffer[h, src_col:src_col + term_dim]
                dst += term_dim
            src_col += term_dim
        return result


# ============================================================================
# PD control
# ============================================================================

def pd_control(target_q, q, kp, dq, kd):
    """Calculate torques from position commands (Humanoid-Gym style)."""
    return (target_q - q) * kp - dq * kd


# ============================================================================
# Policy loading
# ============================================================================

def load_policy(path):
    """Load policy from raw rsl_rl checkpoint or JIT exported model.

    Args:
        path: Path to .pt file (raw checkpoint or TorchScript JIT)

    Returns:
        Callable: policy(obs_np) -> action_np
    """
    import torch
    import torch.nn as nn

    # Try JIT first
    try:
        model = torch.jit.load(path, map_location="cpu")
        model.eval()
        print(f"[INFO] Loaded JIT policy from {path}")

        def jit_policy(obs):
            with torch.no_grad():
                return model(torch.from_numpy(obs).float().unsqueeze(0)).numpy().flatten()
        return jit_policy
    except Exception:
        pass

    # Raw rsl_rl checkpoint
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    # Build actor network matching rsl_rl architecture
    class Actor(nn.Module):
        def __init__(self):
            super().__init__()
            self.actor = nn.Sequential(
                nn.Linear(235, 512), nn.ELU(),
                nn.Linear(512, 256), nn.ELU(),
                nn.Linear(256, 128), nn.ELU(),
                nn.Linear(128, 12),
            )
        def forward(self, x):
            return self.actor(x)

    model = Actor()
    state = {}
    for k, v in ckpt["model_state_dict"].items():
        if k.startswith("actor."):
            state[k] = v
    model.load_state_dict(state)
    model.eval()
    print(f"[INFO] Loaded raw checkpoint from {path}")

    def raw_policy(obs):
        with torch.no_grad():
            return model(torch.from_numpy(obs).float().unsqueeze(0)).numpy().flatten()
    return raw_policy


# ============================================================================
# Keyboard controller
# ============================================================================

class KeyboardController:
    """Keyboard velocity command controller (Humanoid-Gym style interactive mode)."""

    def __init__(self):
        self.vel_cmd = np.array([0.0, 0.0, 0.0])
        self.vel_step = 0.1
        self.running = True
        print("[KEYBOARD] W/S=fwd/back, A/D=yaw, Q/E=lateral, Space=stop, Esc=quit")

    def update(self):
        try:
            import sys, select
            if select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.readline().strip()
                mapping = {'w': (0, 1), 's': (0, -1), 'a': (2, -1), 'd': (2, 1),
                           'q': (1, 1), 'e': (1, -1)}
                if key in mapping:
                    idx, sign = mapping[key]
                    self.vel_cmd[idx] += sign * self.vel_step
                    self.vel_cmd[0] = np.clip(self.vel_cmd[0], -0.5, 1.0)
                    self.vel_cmd[1] = np.clip(self.vel_cmd[1], -0.5, 0.5)
                    self.vel_cmd[2] = np.clip(self.vel_cmd[2], -0.5, 0.5)
                elif key == ' ':
                    self.vel_cmd[:] = 0.0
        except Exception:
            pass


# ============================================================================
# Resolved robot indices (populated at init)
# ============================================================================

class ResolvedRobot:
    """Holds resolved MuJoCo joint/actuator addresses."""
    pass


# ============================================================================
# Main simulation loop (Humanoid-Gym style: run_mujoco(policy, cfg))
# ============================================================================

def run_mujoco(policy, sim_cfg, robot_cfg, obs_cfg, args):
    """Run MuJoCo sim2sim loop with the given policy.

    This follows the Humanoid-Gym pattern: single function drives the sim.
    """
    import mujoco

    # Load model
    model = mujoco.MjModel.from_xml_path(args.mjcf)
    model.opt.timestep = sim_cfg.dt
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)

    # Resolve joint/actuator indices
    robot = ResolvedRobot()
    robot.joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
                       for n in robot_cfg.joint_names]
    robot.actuator_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                          for n in robot_cfg.actuator_names]
    robot.qpos_addr = [model.jnt_qposadr[j] for j in robot.joint_ids]
    robot.dof_addr = [model.jnt_dofadr[j] for j in robot.joint_ids]
    robot.default_joint_pos = robot_cfg.default_joint_pos

    # Sim2sim corrections (match Isaac Lab physics)
    for jid in robot.joint_ids:
        model.dof_damping[jid] = 0.0
    for i, jid in enumerate(robot.joint_ids):
        model.dof_armature[jid] = robot_cfg.armature[i]

    # Observation buffer
    obs_buf = ObservationBuffer(obs_cfg.frame_stack, obs_cfg.num_single_obs, obs_cfg.term_dims)

    # Velocity command
    vel_cmd = np.array([args.vel_x, args.vel_y, args.vel_yaw])
    kb = None
    if args.keyboard:
        kb = KeyboardController()

    # Compute total steps
    duration = args.duration if args.duration else sim_cfg.default_duration
    num_control_steps = int(duration / sim_cfg.control_dt)
    total_physics_steps = int(duration / sim_cfg.dt)

    # --- Reset helper (matches working mujoco_record_video.py exactly) ---
    def reset_sim():
        mujoco.mj_resetData(model, data)
        data.qpos[2] = robot_cfg.init_height
        data.qpos[3] = 1.0  # quat w
        for i, addr in enumerate(robot.qpos_addr):
            data.qpos[addr] = robot_cfg.default_joint_pos[i]
        mujoco.mj_forward(model, data)
        # Warm-up: start with empty buffer, push obs BEFORE physics (old script pattern)
        obs_buf.buffer[:] = 0.0
        zero_act = np.zeros(robot_cfg.num_actions)
        for _ in range(obs_cfg.frame_stack):
            ob = get_obs(data, robot, obs_cfg, zero_act, vel_cmd, 0.0)
            obs_buf.append(ob)
            tgt = robot_cfg.default_joint_pos
            cp = np.array([data.qpos[a] for a in robot.qpos_addr])
            cv = np.array([data.qvel[a] for a in robot.dof_addr])
            tau = np.clip(pd_control(tgt, cp, robot_cfg.kps, cv, robot_cfg.kds),
                         -robot_cfg.tau_limit, robot_cfg.tau_limit)
            for i, aid in enumerate(robot.actuator_ids):
                data.ctrl[aid] = tau[i]
            for _ in range(sim_cfg.decimation):
                mujoco.mj_step(model, data)
        return np.zeros(robot_cfg.num_actions)

    # Initial reset
    last_action = reset_sim()

    # --- Setup renderer or viewer ---
    renderer = None
    viewer = None
    frames = []

    if args.record:
        # EGL offscreen rendering (env var set at module top)
        renderer = mujoco.Renderer(model, height=480, width=640)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, cam)
        cam.distance = 3.0
        cam.elevation = -20
        cam.azimuth = 90
        print(f"[INFO] EGL offscreen recording -> {args.record}")
    elif not args.headless:
        try:
            viewer = mujoco.viewer.launch_passive(model, data)
            print("[INFO] Viewer launched (close window to end)")
        except Exception:
            print("[WARNING] Could not launch viewer, running headless")

    # --- Main loop (matches working mujoco_record_video.py pattern) ---
    # Use decimated control: compute PD torque once, hold for decimation steps
    print(f"[INFO] Simulating {duration:.0f}s ({num_control_steps} control steps @ {1/sim_cfg.control_dt:.0f}Hz)...")
    print(f"[INFO] Velocity command: ({vel_cmd[0]:.2f}, {vel_cmd[1]:.2f}, {vel_cmd[2]:.2f})")

    fall_count = 0
    sim_time = 0.0

    for step in range(num_control_steps):
        # Update velocity command from keyboard
        if kb:
            kb.update()
            vel_cmd = kb.vel_cmd.copy()

        # Fall detection
        if data.qpos[2] < 0.3 or (data.qpos[4]**2 + data.qpos[5]**2 + data.qpos[6]**2) > 0.5:
            fall_count += 1
            last_action = reset_sim()
            sim_time = 0.0
            continue

        # Build observation
        obs_frame = get_obs(data, robot, obs_cfg, last_action, vel_cmd, sim_time)
        obs_buf.append(obs_frame)
        obs_history = obs_buf.get()

        # Policy inference
        action = policy(obs_history)
        action = np.clip(action, -obs_cfg.clip_actions, obs_cfg.clip_actions)
        last_action = action.copy()

        # PD control: compute torque once at control rate
        target_q = robot_cfg.default_joint_pos + action * obs_cfg.action_scale
        cp = np.array([data.qpos[a] for a in robot.qpos_addr])
        cv = np.array([data.qvel[a] for a in robot.dof_addr])
        tau = np.clip(pd_control(target_q, cp, robot_cfg.kps, cv, robot_cfg.kds),
                     -robot_cfg.tau_limit, robot_cfg.tau_limit)
        for i, aid in enumerate(robot.actuator_ids):
            data.ctrl[aid] = tau[i]

        # Physics: decimation steps with held torque
        for _ in range(sim_cfg.decimation):
            mujoco.mj_step(model, data)
        sim_time += sim_cfg.control_dt

        # Render frame
        if renderer:
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render().copy())
        elif viewer:
            try:
                viewer.sync()
            except Exception:
                break

        # Print status every 5 seconds
        if step % int(5.0 / sim_cfg.control_dt) == 0:
            x, y, z = data.qpos[0], data.qpos[1], data.qpos[2]
            print(f"  Step {step:5d} | t={sim_time:.1f}s | pos=({x:.2f},{y:.2f},{z:.2f}) | falls={fall_count}")

        # Real-time pacing for viewer mode
        if viewer and not args.record:
            time.sleep(max(0, sim_cfg.control_dt - 0.001))

    # --- Save video ---
    if args.record and frames:
        import imageio
        print(f"[INFO] Saving {len(frames)} frames to {args.record}...")
        imageio.mimwrite(args.record, frames, fps=int(1.0 / sim_cfg.control_dt))
        sz = os.path.getsize(args.record) / (1024 * 1024)
        print(f"[INFO] Done! Video: {args.record} ({sz:.1f} MB)")
        print(f"[INFO] Falls: {fall_count}")

    if viewer:
        try:
            viewer.close()
        except Exception:
            pass

    print(f"[INFO] Simulation ended. Total time: {sim_time:.1f}s, Falls: {fall_count}")


# ============================================================================
# Entry point
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Humanoid-Gym style MuJoCo sim2sim for MagicBot Z1")

    # Simulator selection
    parser.add_argument("--sim", type=str, default="MuJoCo", choices=["MuJoCo"],
                        help="Simulation backend (default: MuJoCo)")

    # Model paths
    parser.add_argument("--mjcf", type=str, required=True,
                        help="Path to MAGICBOTZ1.xml")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to policy checkpoint (.pt raw or JIT)")

    # Velocity commands
    parser.add_argument("--vel_x", type=float, default=0.5, help="Forward velocity (m/s)")
    parser.add_argument("--vel_y", type=float, default=0.0, help="Lateral velocity (m/s)")
    parser.add_argument("--vel_yaw", type=float, default=0.0, help="Yaw rate (rad/s)")
    parser.add_argument("--keyboard", action="store_true", help="Keyboard velocity control")

    # Recording
    parser.add_argument("--record", type=str, default=None,
                        help="Record video to this path (EGL offscreen, no X Server needed)")
    parser.add_argument("--duration", type=float, default=None,
                        help="Simulation duration in seconds (default: 60)")

    # Display
    parser.add_argument("--headless", action="store_true",
                        help="Run without viewer")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.sim == "MuJoCo":
        policy = load_policy(args.checkpoint)
        run_mujoco(policy, Z1SimConfig(), Z1RobotConfig(), Z1ObsConfig(), args)
    else:
        print(f"[ERROR] Unsupported simulator: {args.sim}")
        sys.exit(1)
