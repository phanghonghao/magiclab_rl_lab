"""Environment configuration file swapper.

Backs up the active ``velocity_env_cfg.py`` and replaces it with the
stage-specific config before launching a new training run.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default location of the active config file (relative to magiclab_rl_lab root)
_DEFAULT_ACTIVE_CFG = (
    "source/magiclab_rl_lab/magiclab_rl_lab/tasks/locomotion/robots/z1/12dof"
    "/velocity_env_cfg.py"
)


class ConfigSwapper:
    """Swap the active environment configuration for a training stage.

    Parameters
    ----------
    project_root:
        Root of the ``magiclab_rl_lab`` project.
    active_cfg_rel:
        Relative path (from *project_root*) to the active config that Isaac Lab
        reads at import time.
    """

    def __init__(
        self,
        project_root: str = ".",
        active_cfg_rel: str = _DEFAULT_ACTIVE_CFG,
    ):
        self._root = Path(project_root).resolve()
        self._active_cfg = self._root / active_cfg_rel
        self._current_backup: Optional[Path] = None

    # -- Swap ---------------------------------------------------------------- #

    def swap(self, stage_config_path: str, stage_id: str) -> str:
        """Backup the current active config and replace it with *stage_config_path*.

        Parameters
        ----------
        stage_config_path:
            Path to the stage-specific config (absolute or relative to CWD).
        stage_id:
            Identifier used for the backup file suffix (e.g. ``"s4_rough_l1"``).

        Returns
        -------
        str
            Absolute path of the backup file (for later :meth:`rollback`).
        """
        src = Path(stage_config_path).resolve()
        if not src.exists():
            raise FileNotFoundError(f"Stage config not found: {src}")
        if not self._active_cfg.exists():
            raise FileNotFoundError(f"Active config not found: {self._active_cfg}")

        # Backup current active config
        backup_path = self._active_cfg.parent / f"{self._active_cfg.name}.bak.{stage_id}"
        shutil.copy2(str(self._active_cfg), str(backup_path))
        self._current_backup = backup_path
        logger.info("Backed up active config → %s", backup_path)

        # Copy stage config into active location (skip if same file)
        if src.samefile(self._active_cfg):
            logger.info("Config already matches stage '%s' — skip swap", stage_id)
        else:
            shutil.copy2(str(src), str(self._active_cfg))
            logger.info("Swapped active config with %s (stage=%s)", src.name, stage_id)

        return str(backup_path)

    # -- Rollback ------------------------------------------------------------ #

    def rollback(self, backup_path: Optional[str] = None) -> None:
        """Restore the previously backed-up config.

        Parameters
        ----------
        backup_path:
            Explicit path to restore from.  If *None*, uses the most recent
            backup from :meth:`swap`.
        """
        if backup_path is not None:
            bp = Path(backup_path)
        elif self._current_backup is not None:
            bp = self._current_backup
        else:
            logger.warning("No backup to rollback to")
            return

        if not bp.exists():
            logger.warning("Backup file not found: %s", bp)
            return

        shutil.copy2(str(bp), str(self._active_cfg))
        logger.info("Rolled back active config from %s", bp.name)
