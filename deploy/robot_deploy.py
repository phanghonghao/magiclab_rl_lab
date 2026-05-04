#!/usr/bin/env python3
"""
MagicBot Z1 Real Robot Locomotion Deployment Script.

Loads the trained locomotion policy and runs it on the real MagicBot Z1 robot
using the low-level SDK interface.

Safety protocol:
    1. Start with suspension test (robot lifted)
    2. Zero velocity commands initially
    3. Gradually increase velocity commands
    4. Emergency stop via Ctrl+C

Usage:
    python deploy/robot_deploy.py \
        --policy path/to/policy.onnx \
        --deploy_cfg path/to/deploy.yaml \
        --robot_ip 192.168.54.111

    # Suspension test mode (robot lifted off ground):
    python deploy/robot_deploy.py \
        --policy policy.onnx \
        --deploy_cfg deploy.yaml \
        --suspension_test

    # With keyboard velocity control:
    python deploy/robot_deploy.py \
        --policy policy.onnx \
        --deploy_cfg deploy.yaml \
        --keyboard
"""

import argparse
import math
import signal
import sys
import threading
import time
import logging

import numpy as np
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default PD gains (matching training config)
DEFAULT_KP = np.array([
    100, 100, 100, 150, 60, 60,
    100, 100, 100, 150, 60, 60,
], dtype=np.float64)

DEFAULT_KD = np.array([
    4, 4, 4, 5, 3, 3,
    4, 4, 4, 5, 3, 3,
], dtype=np.float64)

DEFAULT_JOINT_POS = np.array([
    -0.35, 0.0, 0.0, 0.7, -0.35, 0.0,
    -0.35, 0.0, 0.0, 0.7, -0.35, 0.0,
], dtype=np.float64)

ACTION_SCALE = 0.25

# Observation scales
OBS_SCALE_ANG_VEL = 0.2
OBS_SCALE_JOINT_VEL = 0.05

# Gait phase
GAIT_PERIOD = 0.6

# Control parameters
PHYSICS_DT = 0.002
DECIMATION = 10
CONTROL_DT = PHYSICS_DT * DECIMATION  # 0.02s = 50Hz

# Observation dimensions
OBS_DIM_PER_FRAME = 47
HISTORY_LENGTH = 5
OBS_DIM_TOTAL = OBS_DIM_PER_FRAME * HISTORY_LENGTH  # 235


# ---------------------------------------------------------------------------
# Thread-safe sensor data
# ---------------------------------------------------------------------------

class SensorData:
    """Thread-safe storage for latest sensor readings."""

    def __init__(self, num_joints=12):
        self.lock = threading.Lock()
        self.ang_vel = np.zeros(3)
        self.quat = np.array([1.0, 0.0, 0.0, 0.0])  # (w, x, y, z)
        self.joint_pos = np.zeros(num_joints)
        self.joint_vel = np.zeros(num_joints)
        self.imu_updated = False
        self.leg_updated = False

    def update_imu(self, ang_vel, quat):
        with self.lock:
            self.ang_vel = np.array(ang_vel, dtype=np.float64)
            self.quat = np.array(quat, dtype=np.float64)
            self.imu_updated = True

    def update_leg(self, joint_pos, joint_vel):
        with self.lock:
            self.joint_pos = np.array(joint_pos, dtype=np.float64)
            self.joint_vel = np.array(joint_vel, dtype=np.float64)
            self.leg_updated = True

    def get(self):
        with self.lock:
            return (
                self.ang_vel.copy(),
                self.quat.copy(),
                self.joint_pos.copy(),
                self.joint_vel.copy(),
            )


# ---------------------------------------------------------------------------
# Policy runner
# ---------------------------------------------------------------------------

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
            obs_in = obs.astype(np.float32).reshape(1, -1)
            return self.session.run(None, {"obs": obs_in})[0].flatten()
        else:
            import torch
            with torch.no_grad():
                obs_t = torch.from_numpy(obs).float().reshape(1, -1)
                return self.model(obs_t).numpy().flatten()


# ---------------------------------------------------------------------------
# Observation buffer
# ---------------------------------------------------------------------------

class ObservationBuffer:
    """Rolling observation history buffer (5 frames)."""

    def __init__(self):
        self.buffer = np.zeros((HISTORY_LENGTH, OBS_DIM_PER_FRAME))

    def reset(self, initial_obs):
        for i in range(HISTORY_LENGTH):
            self.buffer[i] = initial_obs

    def append(self, obs):
        self.buffer = np.roll(self.buffer, -1, axis=0)
        self.buffer[-1] = obs
        return self.buffer.flatten()

    def get(self):
        return self.buffer.flatten()


# ---------------------------------------------------------------------------
# Math utilities
# ---------------------------------------------------------------------------

def quat_to_rot_matrix(quat_wxyz):
    """Quaternion (w,x,y,z) to 3x3 rotation matrix."""
    w, x, y, z = quat_wxyz
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z),  2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),       1 - 2*(x*x + y*y)],
    ])


def compute_gait_phase(sim_time, vel_cmd, period=GAIT_PERIOD):
    """Compute gait phase matching training mdp.gait_phase."""
    cmd_norm = np.linalg.norm(vel_cmd)
    if cmd_norm < 0.02:
        return np.array([1.0, 1.0])
    phase = (sim_time % period) / period
    sin_pos = math.sin(2.0 * math.pi * phase)
    return np.array([
        1.0 if sin_pos >= 0 else 0.0,
        1.0 if sin_pos < 0 else 0.0,
    ])


# ---------------------------------------------------------------------------
# Keyboard controller
# ---------------------------------------------------------------------------

class KeyboardController:
    """Keyboard velocity command controller for real-time control."""

    def __init__(self):
        self.vel_cmd = np.array([0.0, 0.0, 0.0])
        self.vel_step = 0.1
        self.running = True
        print("=" * 60)
        print("Keyboard control:")
        print("  W/S = forward/backward")
        print("  A/D = turn left/right")
        print("  Q/E = lateral left/right")
        print("  Space = stop")
        print("  Esc/Ctrl+C = emergency stop")
        print("=" * 60)

    def update(self):
        try:
            import msvcrt
            if msvcrt.kbhit():
                key = msvcrt.getch()
                self._handle(key)
        except ImportError:
            import sys, select
            if select.select([sys.stdin], [], [], 0)[0]:
                line = sys.stdin.readline().strip().lower()
                self._handle_unix(line)

    def _handle(self, key):
        if key == b'w': self.vel_cmd[0] = min(self.vel_cmd[0] + self.vel_step, 1.0)
        elif key == b's': self.vel_cmd[0] = max(self.vel_cmd[0] - self.vel_step, -0.5)
        elif key == b'a': self.vel_cmd[2] = max(self.vel_cmd[2] - self.vel_step, -0.5)
        elif key == b'd': self.vel_cmd[2] = min(self.vel_cmd[2] + self.vel_step, 0.5)
        elif key == b'q': self.vel_cmd[1] = min(self.vel_cmd[1] + self.vel_step, 0.5)
        elif key == b'e': self.vel_cmd[1] = max(self.vel_cmd[1] - self.vel_step, -0.5)
        elif key == b' ': self.vel_cmd[:] = 0.0
        elif key == b'\x1b': self.running = False

    def _handle_unix(self, key):
        if key == 'w': self.vel_cmd[0] = min(self.vel_cmd[0] + self.vel_step, 1.0)
        elif key == 's': self.vel_cmd[0] = max(self.vel_cmd[0] - self.vel_step, -0.5)
        elif key == 'a': self.vel_cmd[2] = max(self.vel_cmd[2] - self.vel_step, -0.5)
        elif key == 'd': self.vel_cmd[2] = min(self.vel_cmd[2] + self.vel_step, 0.5)
        elif key == ' ': self.vel_cmd[:] = 0.0


# ---------------------------------------------------------------------------
# Robot deployment
# ---------------------------------------------------------------------------

class RobotDeploy:
    """Real robot deployment controller."""

    def __init__(self, policy_runner, deploy_cfg, sensor_data):
        self.policy = policy_runner
        self.sensor = sensor_data

        # Load config
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

        # State
        self.vel_cmd = np.array([0.0, 0.0, 0.0])
        self.last_action = np.zeros(12)
        self.obs_buffer = ObservationBuffer()
        self.sim_time = 0.0
        self.running = False

    def build_obs_frame(self):
        """Build a single observation frame (47 dim) from sensor data."""
        ang_vel, quat, joint_pos, joint_vel = self.sensor.get()
        obs = np.zeros(OBS_DIM_PER_FRAME)
        idx = 0

        # 1. Angular velocity in body frame (3), scale=0.2
        # SDK IMU provides angular velocity in body frame already
        obs[idx:idx+3] = ang_vel * OBS_SCALE_ANG_VEL
        idx += 3

        # 2. Projected gravity in body frame (3)
        R = quat_to_rot_matrix(quat)
        gravity_body = R.T @ np.array([0.0, 0.0, -1.0])
        obs[idx:idx+3] = gravity_body
        idx += 3

        # 3. Velocity commands (3)
        obs[idx:idx+3] = self.vel_cmd
        idx += 3

        # 4. Joint positions relative to default (12)
        obs[idx:idx+12] = joint_pos - self.default_joint_pos
        idx += 12

        # 5. Joint velocities (12), scale=0.05
        obs[idx:idx+12] = joint_vel * OBS_SCALE_JOINT_VEL
        idx += 12

        # 6. Last action (12)
        obs[idx:idx+12] = self.last_action
        idx += 12

        # 7. Gait phase (2)
        obs[idx:idx+2] = compute_gait_phase(self.sim_time, self.vel_cmd)
        idx += 2

        return obs

    def compute_joint_targets(self, action):
        """Compute target joint positions from policy action."""
        return self.action_offset + action * self.action_scale

    def initialize_buffer(self):
        """Initialize observation buffer with current state."""
        obs = self.build_obs_frame()
        self.obs_buffer.reset(obs)
        self.last_action = np.zeros(12)
        self.sim_time = 0.0

    def policy_step(self):
        """Run one policy inference step and return target joint positions."""
        obs_history = self.obs_buffer.get()
        action = self.policy.predict(obs_history)
        self.last_action = action.copy()

        target_pos = self.compute_joint_targets(action)

        # Update buffer
        self.sim_time += CONTROL_DT
        new_obs = self.build_obs_frame()
        self.obs_buffer.append(new_obs)

        return target_pos


def parse_args():
    parser = argparse.ArgumentParser(description="MagicBot Z1 real robot deployment")
    parser.add_argument("--policy", type=str, required=True, help="Path to policy (.pt or .onnx)")
    parser.add_argument("--deploy_cfg", type=str, default=None, help="Path to deploy.yaml")
    parser.add_argument("--robot_ip", type=str, default="192.168.54.111", help="Robot IP address")
    parser.add_argument("--onnx", action="store_true", help="Use ONNX model")
    parser.add_argument("--keyboard", action="store_true", help="Use keyboard for velocity commands")
    parser.add_argument("--suspension_test", action="store_true",
                        help="Suspension test mode (robot lifted, zero commands)")
    parser.add_argument("--vel_x", type=float, default=0.0, help="Initial forward velocity (m/s)")
    parser.add_argument("--vel_y", type=float, default=0.0, help="Initial lateral velocity (m/s)")
    parser.add_argument("--vel_yaw", type=float, default=0.0, help="Initial yaw velocity (rad/s)")
    return parser.parse_args()


def main():
    args = parse_args()

    import magicbot_z1_python as magicbot

    # Load deploy config
    deploy_cfg = None
    if args.deploy_cfg:
        with open(args.deploy_cfg, "r") as f:
            deploy_cfg = yaml.safe_load(f)
        logging.info("Loaded deploy config from %s", args.deploy_cfg)

    # Create policy runner
    policy = PolicyRunner(args.policy, use_onnx=args.onnx)
    logging.info("Loaded policy from %s (ONNX=%s)", args.policy, args.onnx)

    # Create sensor data
    sensor = SensorData(num_joints=12)

    # Setup SDK callbacks
    def body_imu_callback(imu_data):
        sensor.update_imu(
            ang_vel=[
                imu_data.angular_velocity[0],
                imu_data.angular_velocity[1],
                imu_data.angular_velocity[2],
            ],
            quat=[
                imu_data.orientation[0],  # w (or x, check SDK convention)
                imu_data.orientation[1],
                imu_data.orientation[2],
                imu_data.orientation[3],
            ],
        )

    def leg_state_callback(joint_state):
        positions = []
        velocities = []
        for i in range(magicbot.LEG_JOINT_NUM):
            positions.append(joint_state.joints[i].posH)
            velocities.append(joint_state.joints[i].vel)
        sensor.update_leg(positions, velocities)

    # Create deployment controller
    deploy = RobotDeploy(policy, deploy_cfg, sensor)
    if not args.suspension_test:
        deploy.vel_cmd = np.array([args.vel_x, args.vel_y, args.vel_yaw])

    # Keyboard controller
    kb_controller = None
    if args.keyboard:
        kb_controller = KeyboardController()

    # Robot initialization
    logging.info("Robot model: %s", magicbot.get_robot_model())
    robot = magicbot.MagicRobot()

    # Graceful shutdown
    running = True

    def signal_handler(signum, frame):
        nonlocal running
        running = False
        logging.info("Emergency stop triggered (signal %s)", signum)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Initialize SDK
        if not robot.initialize(args.robot_ip):
            logging.error("Failed to initialize robot SDK")
            return -1

        # Connect
        status = robot.connect()
        if status.code != magicbot.ErrorCode.OK:
            logging.error("Failed to connect: %s - %s", status.code, status.message)
            robot.shutdown()
            return -1
        logging.info("Connected to robot at %s", args.robot_ip)

        # Switch to low-level control
        status = robot.set_motion_control_level(magicbot.ControllerLevel.LowLevel)
        if status.code != magicbot.ErrorCode.OK:
            logging.error("Failed to switch to low-level: %s - %s", status.code, status.message)
            robot.shutdown()
            return -1
        logging.info("Switched to low-level motion controller")

        controller = robot.get_low_level_motion_controller()

        # Subscribe to sensor data
        controller.subscribe_body_imu(body_imu_callback)
        controller.subscribe_leg_state(leg_state_callback)
        logging.info("Subscribed to IMU and leg state")

        # Wait for sensor data
        logging.info("Waiting for sensor data...")
        timeout = 5.0
        t0 = time.time()
        while not (sensor.imu_updated and sensor.leg_updated):
            time.sleep(0.01)
            if time.time() - t0 > timeout:
                logging.error("Timeout waiting for sensor data")
                robot.disconnect()
                robot.shutdown()
                return -1
        logging.info("Sensor data received")

        # Initialize observation buffer
        deploy.initialize_buffer()

        # Suspension test mode: hold default pose
        if args.suspension_test:
            logging.info("=" * 60)
            logging.info("SUSPENSION TEST MODE")
            logging.info("Robot should be lifted off ground.")
            logging.info("Joints will hold default standing pose.")
            logging.info("Press Ctrl+C to stop.")
            logging.info("=" * 60)

            while running:
                leg_command = magicbot.JointCommand()
                for i in range(magicbot.LEG_JOINT_NUM):
                    joint = magicbot.SingleJointCommand()
                    joint.operation_mode = 200
                    joint.pos = deploy.default_joint_pos[i]
                    joint.vel = 0.0
                    joint.toq = 0.0
                    joint.kp = deploy.kp[i]
                    joint.kd = deploy.kd[i]
                    leg_command.joints.append(joint)
                controller.publish_leg_command(leg_command)
                time.sleep(0.002)

            logging.info("Suspension test ended")
        else:
            # Walking mode
            logging.info("=" * 60)
            logging.info("WALKING MODE")
            logging.info("Robot will start with zero velocity command.")
            if kb_controller:
                logging.info("Use keyboard to control velocity.")
            else:
                logging.info("Velocity command: (%.2f, %.2f, %.2f)",
                             args.vel_x, args.vel_y, args.vel_yaw)
            logging.info("Press Ctrl+C for emergency stop.")
            logging.info("=" * 60)

            # Control loop
            # Run at 500Hz (2ms), update policy every 10 steps (50Hz)
            interval = 0.002
            policy_counter = 0
            target_pos = deploy.default_joint_pos.copy()
            next_t = time.perf_counter() + interval

            while running:
                # Policy update at 50Hz
                if policy_counter % DECIMATION == 0:
                    # Update velocity command from keyboard
                    if kb_controller:
                        kb_controller.update()
                        deploy.vel_cmd = kb_controller.vel_cmd.copy()
                        if not kb_controller.running:
                            break

                    target_pos = deploy.policy_step()

                # Send joint command at 500Hz
                leg_command = magicbot.JointCommand()
                for i in range(magicbot.LEG_JOINT_NUM):
                    joint = magicbot.SingleJointCommand()
                    joint.operation_mode = 200
                    joint.pos = float(target_pos[i])
                    joint.vel = 0.0
                    joint.toq = 0.0
                    joint.kp = float(deploy.kp[i])
                    joint.kd = float(deploy.kd[i])
                    leg_command.joints.append(joint)
                controller.publish_leg_command(leg_command)

                policy_counter += 1

                # Timing
                next_t += interval
                sleep_time = next_t - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)

                # Log every 5 seconds
                if policy_counter % 2500 == 0:
                    ang_vel, quat, jpos, jvel = sensor.get()
                    logging.info(
                        "t=%.1fs | cmd=(%.2f,%.2f,%.2f) | pos_z_est=%.3f",
                        deploy.sim_time,
                        deploy.vel_cmd[0], deploy.vel_cmd[1], deploy.vel_cmd[2],
                        0.0,  # height estimate not available without state estimator
                    )

    except Exception as e:
        logging.error("Exception: %s", e)
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        logging.info("Cleaning up...")
        try:
            # Send zero-position commands for safety
            if 'controller' in dir():
                for _ in range(100):
                    leg_command = magicbot.JointCommand()
                    for i in range(magicbot.LEG_JOINT_NUM):
                        joint = magicbot.SingleJointCommand()
                        joint.operation_mode = 200
                        joint.pos = 0.0
                        joint.vel = 0.0
                        joint.toq = 0.0
                        joint.kp = 20.0  # low gain for safety
                        joint.kd = 2.0
                        leg_command.joints.append(joint)
                    controller.publish_leg_command(leg_command)
                    time.sleep(0.002)

            controller = robot.get_low_level_motion_controller()
            controller.shutdown()
            robot.disconnect()
            robot.shutdown()
            logging.info("Robot shutdown complete")
        except Exception as e:
            logging.error("Cleanup error: %s", e)

    return 0


if __name__ == "__main__":
    sys.exit(main())
