#!/usr/bin/env python3
"""
MuJoCo Sim-to-Sim Validation for MagicBot Z1 Locomotion Policy.

Loads the trained policy (JIT or ONNX) along with deploy.yaml config
and runs the policy in a MuJoCo simulation for validation before real robot deployment.

Usage:
    # With deploy.yaml from training output:
    python sim2sim/mujoco_deploy.py \
        --mjcf ../magicbot-z1_description/mjcf/MAGICBOTZ1.xml \
        --policy logs/rsl_rl/.../exported/policy.pt \
        --deploy_cfg logs/rsl_rl/.../params/deploy.yaml

    # With ONNX model:
    python sim2sim/mujoco_deploy.py \
        --mjcf ../magicbot-z1_description/mjcf/MAGICBOTZ1.xml \
        --policy logs/rsl_rl/.../exported/policy.onnx \
        --deploy_cfg logs/rsl_rl/.../params/deploy.yaml \
        --onnx

    # With keyboard velocity control:
    python sim2sim/mujoco_deploy.py \
        --mjcf ../magicbot-z1_description/mjcf/MAGICBOTZ1.xml \
        --policy policy.pt \
        --deploy_cfg deploy.yaml \
        --keyboard
"""

import argparse
import math
import os
import time

# MUST set before importing mujoco for EGL offscreen rendering
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="MuJoCo sim-to-sim deployment for MagicBot Z1")
    parser.add_argument("--mjcf", type=str, required=True, help="Path to MAGICBOTZ1.xml")
    parser.add_argument("--policy", type=str, required=True, help="Path to policy file (.pt or .onnx)")
    parser.add_argument("--deploy_cfg", type=str, default=None, help="Path to deploy.yaml (optional)")
    parser.add_argument("--onnx", action="store_true", help="Use ONNX model instead of JIT")
    parser.add_argument("--vel_x", type=float, default=0.5, help="Forward velocity command (m/s)")
    parser.add_argument("--vel_y", type=float, default=0.0, help="Lateral velocity command (m/s)")
    parser.add_argument("--vel_yaw", type=float, default=0.0, help="Yaw velocity command (rad/s)")
    parser.add_argument("--keyboard", action="store_true", help="Use keyboard for velocity commands")
    parser.add_argument("--num_steps", type=int, default=10000, help="Number of control steps")
    parser.add_argument("--record", type=str, default=None, help="Record video to this path (EGL offscreen)")
    parser.add_argument("--show_viewer", action="store_true", default=True, help="Show MuJoCo viewer")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# MuJoCo leg joint names in training order
# ---------------------------------------------------------------------------
LEG_JOINTS_MJCF = [
    "JOINT_HIP_PITCH_L", "JOINT_HIP_ROLL_L", "JOINT_HIP_YAW_L",
    "JOINT_KNEE_PITCH_L", "JOINT_ANKLE_PITCH_L", "JOINT_ANKLE_ROLL_L",
    "JOINT_HIP_PITCH_R", "JOINT_HIP_ROLL_R", "JOINT_HIP_YAW_R",
    "JOINT_KNEE_PITCH_R", "JOINT_ANKLE_PITCH_R", "JOINT_ANKLE_ROLL_R",
]

LEG_ACTUATOR_NAMES = [
    "left_hip_pitch_actuator", "left_hip_roll_actuator", "left_hip_yaw_actuator",
    "left_knee_actuator", "left_ankle_pitch_actuator", "left_ankle_roll_actuator",
    "right_hip_pitch_actuator", "right_hip_roll_actuator", "right_hip_yaw_actuator",
    "right_knee_actuator", "right_ankle_pitch_actuator", "right_ankle_roll_actuator",
]

# Default PD gains
# Kp matches training config in magiclab.py exactly
DEFAULT_KP = np.array([
    100, 100, 100, 150, 60, 60,   # left: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
    100, 100, 100, 150, 60, 60,   # right
], dtype=np.float64)

# Kd boosted ~30% vs training values to compensate for implicit PD's built-in
# numerical damping that explicit PD lacks (Isaac Lab IdealPDActuator is implicit)
DEFAULT_KD = np.array([
    5.2, 5.2, 5.2, 6.5, 3.9, 3.9,
    5.2, 5.2, 5.2, 6.5, 3.9, 3.9,
], dtype=np.float64)

# Default joint positions (matching init_state in magiclab.py)
DEFAULT_JOINT_POS = np.array([
    -0.35, 0.0, 0.0, 0.7, -0.35, 0.0,   # left leg
    -0.35, 0.0, 0.0, 0.7, -0.35, 0.0,   # right leg
], dtype=np.float64)

# Action scale from training config
ACTION_SCALE = 0.25

# Observation scales
OBS_SCALE_ANG_VEL = 0.2
OBS_SCALE_JOINT_VEL = 0.05

# Gait phase period (seconds)
GAIT_PERIOD = 0.6

# Simulation parameters (matching training)
PHYSICS_DT = 0.002
DECIMATION = 10
CONTROL_DT = PHYSICS_DT * DECIMATION  # 0.02s = 50Hz

# Contact parameters calibrated for PhysX match
# Friction: use middle of training randomization range (0.3–1.0)
CONTACT_FOOT_FRICTION = (0.65, 0.02, 0.02)     # (slide, torsion, rolling)
CONTACT_GROUND_FRICTION = (0.65, 0.02, 0.02)
# solref: stiffer than MJCF default (-500 -800) to approximate PhysX rigid contact
CONTACT_FOOT_SOLREF = (-3000, -300)
CONTACT_GROUND_SOLREF = (-3000, -300)
# solimp: higher dmin/dmax = stiffer constraint at contact
CONTACT_FOOT_SOLIMP = (0.9, 0.99, 0.001, 0.5, 2)
CONTACT_GROUND_SOLIMP = (0.9, 0.99, 0.001, 0.5, 2)

# Observation dimensions per frame: ang_vel(3) + gravity(3) + cmd(3) + joint_pos(12) + joint_vel(12) + last_action(12) + gait(2)
OBS_DIM_PER_FRAME = 47
HISTORY_LENGTH = 5
OBS_DIM_TOTAL = OBS_DIM_PER_FRAME * HISTORY_LENGTH  # 235

# Per-term dimensions within a single frame (for Isaac Lab compatible history layout)
# Isaac Lab concatenates history per-term: [ang_vel×5, gravity×5, cmd×5, jpos×5, jvel×5, act×5, gait×5]
# NOT frame-stacked: [frame0, frame1, frame2, frame3, frame4]
TERM_DIMS = [3, 3, 3, 12, 12, 12, 2]  # ang_vel, gravity, cmd, jpos, jvel, act, gait
NUM_TERMS = len(TERM_DIMS)


class PolicyRunner:
    """Loads and runs the trained locomotion policy."""

    def __init__(self, policy_path, use_onnx=False):
        self.use_onnx = use_onnx

        if use_onnx:
            import onnxruntime as ort
            self.session = ort.InferenceSession(policy_path)
        else:
            import torch
            self.model = torch.jit.load(policy_path, map_location="cpu")
            self.model.eval()

    def predict(self, obs: np.ndarray) -> np.ndarray:
        if self.use_onnx:
            obs_input = obs.astype(np.float32).reshape(1, -1)
            return self.session.run(None, {"obs": obs_input})[0].flatten()
        else:
            import torch
            with torch.no_grad():
                obs_t = torch.from_numpy(obs).float().reshape(1, -1)
                return self.model(obs_t).numpy().flatten()


class ObservationBuffer:
    """Rolling observation history buffer with Isaac Lab compatible layout.

    Isaac Lab stores history per-term: [term0_hist×5, term1_hist×5, ...]
    NOT frame-stacked: [frame0_all, frame1_all, ...]

    This class stores frames internally, then rearranges to per-term layout on get().
    """

    def __init__(self):
        self.buffer = np.zeros((HISTORY_LENGTH, OBS_DIM_PER_FRAME))

    def reset(self, initial_obs: np.ndarray):
        for i in range(HISTORY_LENGTH):
            self.buffer[i] = initial_obs

    def append(self, obs: np.ndarray) -> np.ndarray:
        self.buffer = np.roll(self.buffer, -1, axis=0)
        self.buffer[-1] = obs
        return self.get()

    def get(self) -> np.ndarray:
        """Return observation in Isaac Lab per-term history layout.

        Converts from [frame0(47), frame1(47), ...] to
        [ang_vel_0..4(15), gravity_0..4(15), ..., gait_0..4(10)]
        """
        # self.buffer shape: (5, 47) — rows are frames, cols are features
        # Rearrange to per-term layout
        result = np.empty(OBS_DIM_TOTAL, dtype=np.float64)
        src_col = 0  # running column offset in each frame
        dst = 0
        for term_dim in TERM_DIMS:
            # For each term, take its columns from all 5 frames
            for h in range(HISTORY_LENGTH):
                result[dst:dst + term_dim] = self.buffer[h, src_col:src_col + term_dim]
                dst += term_dim
            src_col += term_dim
        return result


class KeyboardController:
    """Simple keyboard velocity command controller."""

    def __init__(self):
        self.vel_cmd = np.array([0.0, 0.0, 0.0])
        self.lin_vel_range = (-0.5, 1.0)
        self.ang_vel_range = (-0.5, 0.5)
        self.vel_step = 0.1
        self.running = True
        print("Keyboard control: W/S=forward/back, A/D=turn, Q/E=lateral, Space=stop, Esc=quit")

    def update(self):
        """Non-blocking keyboard check."""
        try:
            import msvcrt
            if msvcrt.kbhit():
                key = msvcrt.getch()
                self._handle_key(key)
        except ImportError:
            # Unix - use select for non-blocking stdin
            import sys, select
            if select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.readline().strip()
                self._handle_key_unix(key)

    def _handle_key(self, key):
        if key == b'w':
            self.vel_cmd[0] = min(self.vel_cmd[0] + self.vel_step, self.lin_vel_range[1])
        elif key == b's':
            self.vel_cmd[0] = max(self.vel_cmd[0] - self.vel_step, self.lin_vel_range[0])
        elif key == b'a':
            self.vel_cmd[2] = max(self.vel_cmd[2] - self.vel_step, self.ang_vel_range[0])
        elif key == b'd':
            self.vel_cmd[2] = min(self.vel_cmd[2] + self.vel_step, self.ang_vel_range[1])
        elif key == b'q':
            self.vel_cmd[1] = min(self.vel_cmd[1] + self.vel_step, 0.5)
        elif key == b'e':
            self.vel_cmd[1] = max(self.vel_cmd[1] - self.vel_step, -0.5)
        elif key == b' ':
            self.vel_cmd[:] = 0.0
        elif key == b'\x1b':
            self.running = False

    def _handle_key_unix(self, key):
        mapping = {'w': 0, 's': 0, 'a': 2, 'd': 2, 'q': 1, 'e': 1}
        if key in mapping:
            idx = mapping[key]
            sign = 1 if key in ('w', 'd', 'q') else -1
            self.vel_cmd[idx] += sign * self.vel_step
            self.vel_cmd[0] = np.clip(self.vel_cmd[0], *self.lin_vel_range)
            self.vel_cmd[1] = np.clip(self.vel_cmd[1], -0.5, 0.5)
            self.vel_cmd[2] = np.clip(self.vel_cmd[2], *self.ang_vel_range)
        elif key == ' ':
            self.vel_cmd[:] = 0.0
        elif key == 'q!':
            self.running = False


def quat_to_rot_matrix(quat_wxyz):
    """Convert quaternion (w,x,y,z) to 3x3 rotation matrix."""
    w, x, y, z = quat_wxyz
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z),  2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),       1 - 2*(x*x + y*y)],
    ])


def compute_gait_phase(sim_time, vel_cmd, period=GAIT_PERIOD):
    """Compute gait phase observation matching training mdp.gait_phase."""
    cmd_norm = np.linalg.norm(vel_cmd)
    if cmd_norm < 0.02:
        return np.array([1.0, 1.0])  # standing: both feet stance

    phase = (sim_time % period) / period
    sin_pos = math.sin(2.0 * math.pi * phase)
    left_stance = 1.0 if sin_pos >= 0 else 0.0
    right_stance = 1.0 if sin_pos < 0 else 0.0
    return np.array([left_stance, right_stance])


class MuJoCoDeploy:
    """MuJoCo sim-to-sim deployment environment."""

    def __init__(self, mjcf_path, policy_runner, deploy_cfg=None, vel_cmd=None):
        import mujoco

        self.policy = policy_runner
        self.vel_cmd = np.array(vel_cmd) if vel_cmd is not None else np.array([0.5, 0.0, 0.0])

        # Load PD gains from deploy_cfg or use defaults
        if deploy_cfg:
            self.kp = np.array(deploy_cfg.get("stiffness", DEFAULT_KP.tolist()))
            self.kd = np.array(deploy_cfg.get("damping", DEFAULT_KD.tolist()))
            self.default_joint_pos = np.array(deploy_cfg.get("default_joint_pos", DEFAULT_JOINT_POS.tolist()))

            action_cfg = deploy_cfg.get("actions", {}).get("JointPositionAction", {})
            self.action_scale = np.array(action_cfg.get("scale", [ACTION_SCALE] * 12))
            self.action_offset = np.array(action_cfg.get("offset", DEFAULT_JOINT_POS.tolist()))
        else:
            self.kp = DEFAULT_KP.copy()
            self.kd = DEFAULT_KD.copy()
            self.default_joint_pos = DEFAULT_JOINT_POS.copy()
            self.action_scale = np.full(12, ACTION_SCALE)
            self.action_offset = DEFAULT_JOINT_POS.copy()

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.model.opt.timestep = PHYSICS_DT
        self.data = mujoco.MjData(self.model)

        # Resolve joint and actuator indices
        self.leg_joint_ids = []
        for name in LEG_JOINTS_MJCF:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            assert jid >= 0, f"Joint '{name}' not found in MJCF model"
            self.leg_joint_ids.append(jid)

        self.leg_actuator_ids = []
        for name in LEG_ACTUATOR_NAMES:
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            assert aid >= 0, f"Actuator '{name}' not found in MJCF model"
            self.leg_actuator_ids.append(aid)

        # qpos/qvel addresses for leg joints
        self.leg_qpos_addr = [self.model.jnt_qposadr[jid] for jid in self.leg_joint_ids]
        self.leg_dof_addr = [self.model.jnt_dofadr[jid] for jid in self.leg_joint_ids]

        # --- Sim2sim corrections (match Isaac Lab physics) ---
        # Clear MJCF default joint damping (Isaac Lab uses 0, damping comes from PD controller)
        for jid in self.leg_joint_ids:
            self.model.dof_damping[jid] = 0.0

        # Add armature to match Isaac Lab ImplicitActuatorCfg
        armature = np.array([0.02863, 0.02863, 0.02863, 0.02863, 0.01503, 0.01503,
                             0.02863, 0.02863, 0.02863, 0.02863, 0.01503, 0.01503])
        for i, jid in enumerate(self.leg_joint_ids):
            self.model.dof_armature[jid] = armature[i]

        # Apply contact parameters calibrated for PhysX match
        for geom_id in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if name in ("l_foot", "r_foot"):
                self.model.geom_friction[geom_id] = CONTACT_FOOT_FRICTION
                self.model.geom_solref[geom_id] = CONTACT_FOOT_SOLREF
                self.model.geom_solimp[geom_id] = CONTACT_FOOT_SOLIMP
            elif name == "ground":
                self.model.geom_friction[geom_id] = CONTACT_GROUND_FRICTION
                self.model.geom_solref[geom_id] = CONTACT_GROUND_SOLREF
                self.model.geom_solimp[geom_id] = CONTACT_GROUND_SOLIMP

        # Effort limits matching training config
        self.effort_limits = np.array([120, 120, 120, 120, 50, 50,
                                       120, 120, 120, 120, 50, 50], dtype=np.float64)

        # State
        self.obs_buffer = ObservationBuffer()
        self.last_action = np.zeros(12)
        self.sim_time = 0.0

        self.reset()

    def reset(self):
        """Reset simulation to initial standing pose."""
        import mujoco
        mujoco.mj_resetData(self.model, self.data)

        # Set pelvis initial height (matching Isaac Lab init_state)
        self.data.qpos[2] = 0.69  # z

        # Quaternion identity (w, x, y, z)
        self.data.qpos[3] = 1.0
        self.data.qpos[4:7] = 0.0

        # Set default leg joint positions
        for i, addr in enumerate(self.leg_qpos_addr):
            self.data.qpos[addr] = self.default_joint_pos[i]

        mujoco.mj_forward(self.model, self.data)

        # Reset buffers
        self.last_action = np.zeros(12)
        self.sim_time = 0.0

        # Warm-up: fill observation history with valid data (not zeros)
        obs = self._build_obs_frame()
        self.obs_buffer.reset(obs)
        for _ in range(HISTORY_LENGTH):
            # Hold default pose for warm-up steps
            tgt = self.default_joint_pos
            for _ in range(DECIMATION):
                cp = np.array([self.data.qpos[a] for a in self.leg_qpos_addr])
                cv = np.array([self.data.qvel[a] for a in self.leg_dof_addr])
                torques = np.clip(self.kp * (tgt - cp) - self.kd * cv, -self.effort_limits, self.effort_limits)
                for i, act_id in enumerate(self.leg_actuator_ids):
                    self.data.ctrl[act_id] = torques[i]
                mujoco.mj_step(self.model, self.data)
            self.sim_time += CONTROL_DT

            obs = self._build_obs_frame()
            self.obs_buffer.append(obs)

    def _get_pelvis_rot(self):
        """Get pelvis rotation matrix from quaternion."""
        quat = self.data.qpos[3:7].copy()  # (w, x, y, z)
        return quat_to_rot_matrix(quat)

    def _build_obs_frame(self):
        """Build a single observation frame (47 dim)."""
        obs = np.zeros(OBS_DIM_PER_FRAME)
        idx = 0

        R_wb = self._get_pelvis_rot()  # world-to-body rotation
        R_bw = R_wb.T                   # body-to-world rotation (MuJoCo convention: xmat is body-to-world)

        # 1. Angular velocity in body frame (3), scale=0.2
        omega_world = self.data.qvel[3:6].copy()
        omega_body = R_wb @ omega_world
        obs[idx:idx+3] = omega_body * OBS_SCALE_ANG_VEL
        idx += 3

        # 2. Projected gravity in body frame (3)
        gravity_world = np.array([0.0, 0.0, -1.0])
        gravity_body = R_wb @ gravity_world
        obs[idx:idx+3] = gravity_body
        idx += 3

        # 3. Velocity commands (3)
        obs[idx:idx+3] = self.vel_cmd
        idx += 3

        # 4. Joint positions relative to default (12)
        joint_pos = np.array([self.data.qpos[a] for a in self.leg_qpos_addr])
        obs[idx:idx+12] = joint_pos - self.default_joint_pos
        idx += 12

        # 5. Joint velocities (12), scale=0.05
        joint_vel = np.array([self.data.qvel[a] for a in self.leg_dof_addr])
        obs[idx:idx+12] = joint_vel * OBS_SCALE_JOINT_VEL
        idx += 12

        # 6. Last action (12)
        obs[idx:idx+12] = self.last_action
        idx += 12

        # 7. Gait phase (2)
        obs[idx:idx+2] = compute_gait_phase(self.sim_time, self.vel_cmd)
        idx += 2

        return obs

    def step(self):
        """Execute one control step (50Hz). Returns True if fall detected."""
        # Policy inference
        obs_history = self.obs_buffer.get()
        action = self.policy.predict(obs_history)
        self.last_action = action.copy()

        # Compute target joint positions: target = offset + action * scale
        target_pos = self.action_offset + action * self.action_scale

        # Step physics with PD recomputed at every sub-step
        # Isaac Lab implicit PD effectively updates every physics step,
        # so recomputing explicit PD each sub-step narrows the sim2sim gap.
        import mujoco
        for _ in range(DECIMATION):
            current_pos = np.array([self.data.qpos[a] for a in self.leg_qpos_addr])
            current_vel = np.array([self.data.qvel[a] for a in self.leg_dof_addr])
            torques = np.clip(
                self.kp * (target_pos - current_pos) - self.kd * current_vel,
                -self.effort_limits, self.effort_limits
            )
            for i, act_id in enumerate(self.leg_actuator_ids):
                self.data.ctrl[act_id] = torques[i]
            mujoco.mj_step(self.model, self.data)

        self.sim_time += CONTROL_DT

        # Fall detection: z < 0.3 or orientation deviates from upright
        # Quaternion (w,x,y,z): upright means w≈1, x≈y≈z≈0
        if self.data.qpos[2] < 0.3 or (self.data.qpos[4]**2 + self.data.qpos[5]**2 + self.data.qpos[6]**2) > 0.5:
            return True

        # Update observation buffer
        new_obs = self._build_obs_frame()
        self.obs_buffer.append(new_obs)
        return False

    def get_robot_state(self):
        """Get current robot state for display."""
        pos = self.data.qpos[:3].copy()
        return {
            "x": pos[0], "y": pos[1], "z": pos[2],
            "time": self.sim_time,
        }


def main():
    args = parse_args()

    # Load deploy config if provided
    deploy_cfg = None
    if args.deploy_cfg:
        with open(args.deploy_cfg, "r") as f:
            deploy_cfg = yaml.safe_load(f)
        print(f"[INFO] Loaded deploy config from {args.deploy_cfg}")

    # Create policy runner
    policy = PolicyRunner(args.policy, use_onnx=args.onnx)
    print(f"[INFO] Loaded policy from {args.policy} (ONNX={args.onnx})")

    # Velocity command
    vel_cmd = [args.vel_x, args.vel_y, args.vel_yaw]
    kb_controller = None
    if args.keyboard:
        kb_controller = KeyboardController()

    # Create MuJoCo environment
    env = MuJoCoDeploy(args.mjcf, policy, deploy_cfg=deploy_cfg, vel_cmd=vel_cmd)
    print(f"[INFO] MuJoCo model loaded from {args.mjcf}")
    print(f"[INFO] Control frequency: {1.0/CONTROL_DT:.0f}Hz, Physics: {1.0/PHYSICS_DT:.0f}Hz")
    print(f"[INFO] Observation: {OBS_DIM_TOTAL}d ({OBS_DIM_PER_FRAME}d x {HISTORY_LENGTH} frames)")
    print(f"[INFO] Velocity command: {vel_cmd}")

    # Launch viewer or renderer
    import mujoco
    viewer = None
    renderer = None
    frames = []

    if args.record:
        renderer = mujoco.Renderer(env.model, height=480, width=640)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(env.model, cam)
        cam.distance = 3.0
        cam.elevation = -20
        cam.azimuth = 90
        print(f"[INFO] EGL offscreen recording -> {args.record}")
        args.show_viewer = False
    elif args.show_viewer:
        try:
            viewer = mujoco.viewer.launch_passive(env.model, env.data)
            print("[INFO] Viewer launched (close window to end)")
        except Exception:
            print("[WARNING] Could not launch MuJoCo viewer, running headless")
            viewer = None

    # Main loop
    fall_count = 0
    print(f"[INFO] Running {args.num_steps} steps...")
    try:
        for step in range(args.num_steps):
            # Update velocity command from keyboard
            if kb_controller:
                kb_controller.update()
                env.vel_cmd = kb_controller.vel_cmd.copy()
                if not kb_controller.running:
                    break

            fell = env.step()
            if fell:
                env.reset()
                fall_count += 1

            # Record frame
            if renderer:
                renderer.update_scene(env.data, camera=cam)
                frames.append(renderer.render().copy())

            # Sync viewer
            if viewer is not None:
                try:
                    viewer.sync()
                except Exception:
                    break

            # Print status every 500 steps (10 seconds at 50Hz)
            if step % 500 == 0:
                state = env.get_robot_state()
                print(f"  Step {step:5d} | t={state['time']:.1f}s | "
                      f"pos=({state['x']:.2f}, {state['y']:.2f}, {state['z']:.2f}) | "
                      f"falls={fall_count}")

            # Sleep for real-time (only in viewer mode)
            if viewer and not args.record:
                time.sleep(max(0, CONTROL_DT - 0.001))

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")

    # Save video
    if args.record and frames:
        import imageio
        print(f"[INFO] Saving {len(frames)} frames to {args.record}...")
        imageio.mimwrite(args.record, frames, fps=int(1.0 / CONTROL_DT))
        sz = os.path.getsize(args.record) / (1024 * 1024)
        print(f"[INFO] Done! {args.record} ({sz:.1f} MB), Falls: {fall_count}")

    if viewer is not None:
        try:
            viewer.close()
        except Exception:
            pass

    print("[INFO] Simulation ended")


if __name__ == "__main__":
    main()
