"""Observations for the Z1 throwing task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ee_position(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """End-effector (palm) position in robot base frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    ee_pos_w = robot.data.body_pos_w[:, asset_cfg.body_ids[0]]
    # Transform to base frame
    base_pos = robot.data.root_pos_w
    base_quat = robot.data.root_quat_w
    from isaaclab.utils.math import quat_rotate_inverse
    ee_pos_b = quat_rotate_inverse(base_quat, ee_pos_w - base_pos)
    return ee_pos_b


def ee_velocity(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """End-effector (palm) linear velocity in robot base frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    ee_vel_w = robot.data.body_lin_vel_w[:, asset_cfg.body_ids[0]]
    base_quat = robot.data.root_quat_w
    from isaaclab.utils.math import quat_rotate_inverse
    ee_vel_b = quat_rotate_inverse(base_quat, ee_vel_w)
    return ee_vel_b


def ball_position(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Ball position in robot base frame."""
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene["robot"]
    ball_pos_w = ball.data.root_pos_w
    base_pos = robot.data.root_pos_w
    base_quat = robot.data.root_quat_w
    from isaaclab.utils.math import quat_rotate_inverse
    ball_pos_b = quat_rotate_inverse(base_quat, ball_pos_w - base_pos)
    return ball_pos_b


def ball_velocity(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Ball velocity in robot base frame."""
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene["robot"]
    ball_vel_w = ball.data.root_lin_vel_w
    base_quat = robot.data.root_quat_w
    from isaaclab.utils.math import quat_rotate_inverse
    ball_vel_b = quat_rotate_inverse(base_quat, ball_vel_w)
    return ball_vel_b


def ball_on_palm_flag(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    threshold: float = 0.10,
) -> torch.Tensor:
    """Binary flag: is the ball on the palm?"""
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[asset_cfg.name]
    ee_pos_w = robot.data.body_pos_w[:, asset_cfg.body_ids[0]]
    ball_pos_w = ball.data.root_pos_w
    dist = torch.norm(ball_pos_w - ee_pos_w, dim=-1, keepdim=True)
    return (dist < threshold).float()


def target_position(
    env: ManagerBasedRLEnv,
    target_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Target basket position in robot base frame."""
    robot: Articulation = env.scene["robot"]
    target_asset = env.scene[target_cfg.name]
    target_pos_w = target_asset.data.root_pos_w[:, :3] if hasattr(target_asset, 'data') else torch.zeros(env.num_envs, 3, device=env.device)
    base_pos = robot.data.root_pos_w
    base_quat = robot.data.root_quat_w
    from isaaclab.utils.math import quat_rotate_inverse
    target_pos_b = quat_rotate_inverse(base_quat, target_pos_w - base_pos)
    return target_pos_b
