"""Rewards for the Z1 throwing task."""

from __future__ import annotations

import torch
import math
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ball_on_palm(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg,
    ee_body_name: str,
    threshold: float = 0.08,
) -> torch.Tensor:
    """Reward for keeping the ball on the palm."""
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene["robot"]
    # Get palm position in world frame
    ee_body_ids, _ = robot.find_bodies(ee_body_name)
    ee_pos = robot.data.body_pos_w[:, ee_body_ids[0]]  # (num_envs, 3)
    ball_pos = ball.data.root_pos_w  # (num_envs, 3)
    dist = torch.norm(ball_pos - ee_pos, dim=-1)
    return (dist < threshold).float()


def ball_distance_to_target(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward based on distance between ball and target basket."""
    ball: RigidObject = env.scene[ball_cfg.name]
    # target is a static asset, read its position from scene
    target_pos = env.scene[target_cfg.name].data.root_pos_w[:, :3] if hasattr(env.scene[target_cfg.name], 'data') else torch.zeros(env.num_envs, 3, device=env.device)
    ball_pos = ball.data.root_pos_w
    dist = torch.norm(ball_pos - target_pos, dim=-1)
    return torch.exp(-dist / 0.5)


def ball_release_velocity_reward(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    ee_body_name: str,
    release_dist: float = 0.12,
) -> torch.Tensor:
    """Reward for ball velocity direction toward target when released from palm."""
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene["robot"]
    ee_body_ids, _ = robot.find_bodies(ee_body_name)
    ee_pos = robot.data.body_pos_w[:, ee_body_ids[0]]
    ball_pos = ball.data.root_pos_w
    dist_to_ee = torch.norm(ball_pos - ee_pos, dim=-1)
    # Only reward when ball just left the palm
    just_released = (dist_to_ee > release_dist * 0.5) & (dist_to_ee < release_dist * 2.0)
    ball_vel = ball.data.root_lin_vel_w
    ball_speed = torch.norm(ball_vel, dim=-1, keepdim=True).clamp(min=1e-6)
    ball_vel_normalized = ball_vel / ball_speed
    target_pos = env.scene[target_cfg.name].data.root_pos_w[:, :3] if hasattr(env.scene[target_cfg.name], 'data') else torch.zeros(env.num_envs, 3, device=env.device)
    direction_to_target = target_pos - ball_pos
    dir_norm = torch.norm(direction_to_target, dim=-1, keepdim=True).clamp(min=1e-6)
    direction_to_target_normalized = direction_to_target / dir_norm
    alignment = torch.sum(ball_vel_normalized * direction_to_target_normalized, dim=-1)
    reward = alignment * ball_speed.squeeze(-1)
    return reward * just_released.float()


def ball_in_basket(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    basket_radius: float = 0.15,
    basket_height_range: tuple = (-0.05, 0.3),
) -> torch.Tensor:
    """Sparse reward for ball entering the basket."""
    ball: RigidObject = env.scene[ball_cfg.name]
    ball_pos = ball.data.root_pos_w
    target_pos = env.scene[target_cfg.name].data.root_pos_w[:, :3] if hasattr(env.scene[target_cfg.name], 'data') else torch.zeros(env.num_envs, 3, device=env.device)
    # Check horizontal distance
    horizontal_dist = torch.norm(ball_pos[:, :2] - target_pos[:, :2], dim=-1)
    # Check vertical: ball should be near or slightly above basket rim
    vertical_ok = (ball_pos[:, 2] > target_pos[:, 2] + basket_height_range[0]) & \
                  (ball_pos[:, 2] < target_pos[:, 2] + basket_height_range[1])
    in_basket = (horizontal_dist < basket_radius) & vertical_ok
    return in_basket.float()


def ball_fall_penalty(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg,
    fall_height: float = 0.3,
) -> torch.Tensor:
    """Penalty for ball falling below threshold."""
    ball: RigidObject = env.scene[ball_cfg.name]
    ball_pos = ball.data.root_pos_w
    return (ball_pos[:, 2] < fall_height).float()


def arm_energy_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize energy used by arm joints."""
    asset: Articulation = env.scene[asset_cfg.name]
    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def joint_limit_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty for joints near their limits."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    soft_limit = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]
    # Distance to nearest limit
    lower_dist = torch.abs(joint_pos - soft_limit[:, :, 0])
    upper_dist = torch.abs(soft_limit[:, :, 1] - joint_pos)
    min_dist = torch.minimum(lower_dist, upper_dist)
    # Penalty when within 10% of limit
    range_size = soft_limit[:, :, 1] - soft_limit[:, :, 0]
    near_limit = (min_dist < range_size * 0.1).float()
    return torch.sum(near_limit, dim=-1)
