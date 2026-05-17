"""Commands for the Z1 throwing task — target position generation."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class UniformTargetCommand:
    """Generate random target positions for the basket."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        self.cfg = cfg
        self.env = env

    def has_any_command(self):
        return True

    def get_command(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """Return the target position command."""
        return env.command_manager.get_command("target_position")

    def get_signaled_command(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """Return the target position command."""
        return env.command_manager.get_command("target_position")


@configclass
class UniformTargetCommandCfg(CommandTermCfg):
    """Configuration for uniform target position command."""

    func = UniformTargetCommand

    @configclass
    class Ranges:
        distance: tuple[float, float] = (1.0, 3.0)
        azimuth: tuple[float, float] = (-0.5, 0.5)  # radians, relative to forward
        height: tuple[float, float] = (1.2, 2.0)

    ranges: Ranges = Ranges()
