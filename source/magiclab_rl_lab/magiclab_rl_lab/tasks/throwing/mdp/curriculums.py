"""Curriculums for the Z1 throwing task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def target_distance_levels(
    env: ManagerBasedRLEnv,
    command_name: str = "target_position",
    start_dist: float = 1.0,
    end_dist: float = 3.0,
) -> torch.Tensor:
    """Curriculum that gradually increases the throwing distance."""
    # Compute mean reward as a progress metric
    mean_reward = torch.mean(torch.sum(env.reward_manager._step_reward, dim=0))
    # Progress from 0 to 1 based on mean reward
    progress = (mean_reward / 10.0).clamp(0.0, 1.0)
    current_max_dist = start_dist + (end_dist - start_dist) * progress
    return current_max_dist.expand(env.num_envs)
