"""Generate velocity_env_cfg.py from a parameter dict.

Reads the merged phase/sub-phase parameters and produces a complete
Python config file that Isaac Lab can import at runtime.  Uses the
original velocity_env_cfg.py as a base template and replaces only the
dynamic sections (terrain, commands, rewards, terminations, sim params).

This avoids f-string brace conflicts with Isaac Lab's ``{ENV_REGEX_NS}``
and other placeholder syntax.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Path to the active config that Isaac Lab reads
_ACTIVE_CFG_REL = (
    "source/magiclab_rl_lab/magiclab_rl_lab/tasks/locomotion"
    "/robots/z1/12dof/velocity_env_cfg.py"
)

# Path to the original template config (used as base)
_TEMPLATE_CFG_REL = _ACTIVE_CFG_REL


# ── Terrain generator builder ──────────────────────────────────── #


def _build_terrain_generator_block(terrain_cfg: Optional[dict]) -> str:
    """Return Python source for the COBBLESTONE_ROAD_CFG variable."""
    if terrain_cfg is None:
        return "COBBLESTONE_ROAD_CFG = None"

    lines = [
        "COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(",
    ]
    for key in ("size", "border_width", "num_rows", "num_cols",
                "horizontal_scale", "vertical_scale", "slope_threshold"):
        if key in terrain_cfg:
            val = terrain_cfg[key]
            lines.append(f"    {key}={val!r},")
    if "difficulty_range" in terrain_cfg:
        lines.append(f"    difficulty_range={terrain_cfg['difficulty_range']!r},")
    lines.append("    use_cache=False,")
    subs = terrain_cfg.get("sub_terrains", {})
    if subs:
        lines.append("    sub_terrains={")
        _type_map = {
            "MeshPlaneTerrainCfg": "terrain_gen.MeshPlaneTerrainCfg",
            "RandomGridTerrainCfg": "terrain_gen.MeshRandomGridTerrainCfg",
            "StairsTerrainCfg": "terrain_gen.MeshPyramidStairsTerrainCfg",
            "GapTerrainCfg": "terrain_gen.MeshGapTerrainCfg",
            "BoxesTerrainCfg": "terrain_gen.MeshBoxTerrainCfg",
        }
        for name, scfg in subs.items():
            cls_name = _type_map.get(scfg.get("type", ""), "terrain_gen.MeshPlaneTerrainCfg")
            parts = [f'proportion={scfg.get("proportion", 0.5)}']
            # Each trimesh terrain type has its own height/width params (not difficulty_range)
            terrain_type = scfg.get("type", "")
            diff = scfg.get("difficulty_range", [0.0, 0.5])
            d_max = diff[1] if len(diff) > 1 else 0.5
            if terrain_type == "RandomGridTerrainCfg":
                parts.append(f'grid_width=0.6')
                parts.append(f'grid_height_range=({diff[0]}, {d_max})')
            elif terrain_type == "StairsTerrainCfg":
                parts.append(f'step_height_range=(0.05, {d_max * 0.25:.2f})')
                parts.append(f'step_width=0.3')
            elif terrain_type == "BoxesTerrainCfg":
                parts.append(f'box_height_range=(0.05, {d_max * 0.3:.2f})')
            elif terrain_type == "GapTerrainCfg":
                parts.append(f'gap_width_range=(0.1, {d_max * 0.5:.2f})')
            elif "difficulty_range" in scfg:
                parts.append(f'difficulty_range={scfg["difficulty_range"]!r}')
            lines.append(f'        "{name}": {cls_name}({", ".join(parts)}),')
        lines.append("    },")
    lines.append(")")
    return "\n".join(lines)


# Event block builder

def _build_events_block(event_cfg: dict) -> str:
    """Return source for EventCfg class.

    Supports disabling terms by setting them to ``null`` in YAML::

        push_robot: null                 → skip push_robot term
        base_external_force_torque: null → skip external force term

    Supports startup/restart customisation via sub-keys::

        startup:
          physics_material:
            friction_range: [0.8, 1.0]
          add_base_mass: null            → skip mass randomisation
          randomize_others_mass: null    → skip
        reset:
          reset_robot_joints:
            velocity_range: [0.0, 0.0]
    """
    startup_cfg = event_cfg.get("startup", {})
    reset_cfg = event_cfg.get("reset", {})

    # ── Physics material ──
    physics_cfg = startup_cfg.get("physics_material", {})
    friction = tuple(physics_cfg.get("friction_range", (0.1, 2.0)))

    # ── Add base mass ──
    base_mass_raw = startup_cfg.get("add_base_mass", "NOT_SET")
    if base_mass_raw is None:
        base_mass_enabled = False
        base_mass_scale = (0.5, 1.5)
    elif isinstance(base_mass_raw, dict):
        base_mass_enabled = True
        base_mass_scale = tuple(base_mass_raw.get("scale_range", (0.5, 1.5)))
    else:
        base_mass_enabled = True
        base_mass_scale = (0.5, 1.5)

    # ── Randomise others mass ──
    others_mass_raw = startup_cfg.get("randomize_others_mass", "NOT_SET")
    if others_mass_raw is None:
        others_mass_enabled = False
        others_mass_scale = (0.5, 1.5)
    elif isinstance(others_mass_raw, dict):
        others_mass_enabled = True
        others_mass_scale = tuple(others_mass_raw.get("scale_range", (0.5, 1.5)))
    else:
        others_mass_enabled = True
        others_mass_scale = (0.5, 1.5)

    # ── Reset robot joints ──
    joints_cfg = reset_cfg.get("reset_robot_joints", {})
    joint_vel_range = tuple(joints_cfg.get("velocity_range", (-1.0, 1.0)))

    # ── External force/torque (reset mode) — null to disable ──
    ext_force_raw = event_cfg.get("base_external_force_torque", "NOT_SET")
    ext_force_enabled = ext_force_raw is not None
    ext_force_dict = ext_force_raw if isinstance(ext_force_raw, dict) else {}
    force_range = tuple(ext_force_dict.get("force_range", (0.0, 0.0)))
    torque_range = tuple(ext_force_dict.get("torque_range", (0.0, 0.0)))

    # ── Interval force/torque ──
    interval_cfg = event_cfg.get("interval_force_torque", {})
    interval_force = tuple(interval_cfg.get("force_range", (0.0, 0.0)))
    interval_torque = tuple(interval_cfg.get("torque_range", (0.0, 0.0)))
    interval_range_s = tuple(interval_cfg.get("interval_range_s", (999.0, 999.0)))

    # ── Reset base ──
    reset_base_cfg = event_cfg.get("reset_base", {})
    pose_range = reset_base_cfg.get(
        "pose_range",
        {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
    )
    velocity_range = reset_base_cfg.get(
        "velocity_range",
        {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "z": (-0.5, 0.5),
            "roll": (-0.5, 0.5),
            "pitch": (-0.5, 0.5),
            "yaw": (-0.5, 0.5),
        },
    )

    # ── Push robot — null to disable ──
    push_raw = event_cfg.get("push_robot", "NOT_SET")
    push_enabled = push_raw is not None
    push_dict = push_raw if isinstance(push_raw, dict) else {}
    push_interval = tuple(push_dict.get("interval_range_s", (3.0, 5.0)))
    push_velocity = push_dict.get(
        "velocity_range", {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}
    )

    # ── Build block ──
    p: list[str] = []

    p.append('@configclass\n')
    p.append('class EventCfg:\n')
    p.append('    """Configuration for events."""\n\n')

    # startup
    p.append('    # startup\n')
    p.append('    physics_material = EventTerm(\n')
    p.append('        func=mdp.randomize_rigid_body_material,\n')
    p.append('        mode="startup",\n')
    p.append('        params={\n')
    p.append('            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),\n')
    p.append(f'            "static_friction_range": {friction!r},\n')
    p.append(f'            "dynamic_friction_range": {friction!r},\n')
    p.append('            "restitution_range": (0.0, 0.0),\n')
    p.append('            "num_buckets": 64,\n')
    p.append('        },\n')
    p.append('    )\n\n')

    if base_mass_enabled:
        p.append('    add_base_mass = EventTerm(\n')
        p.append('        func=mdp.randomize_rigid_body_mass,\n')
        p.append('        mode="startup",\n')
        p.append('        params={\n')
        p.append('            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),\n')
        p.append(f'            "mass_distribution_params": {base_mass_scale!r},\n')
        p.append('            "operation": "scale",\n')
        p.append('            "recompute_inertia": True,\n')
        p.append('        },\n')
        p.append('    )\n\n')

    if others_mass_enabled:
        p.append('    randomize_rigid_body_mass_others = EventTerm(\n')
        p.append('        func=mdp.randomize_rigid_body_mass,\n')
        p.append('        mode="startup",\n')
        p.append('        params={\n')
        p.append('            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),\n')
        p.append(f'            "mass_distribution_params": {others_mass_scale!r},\n')
        p.append('            "operation": "scale",\n')
        p.append('            "recompute_inertia": True,\n')
        p.append('        },\n')
        p.append('    )\n\n')

    # reset
    p.append('    # reset\n')

    if ext_force_enabled:
        p.append('    base_external_force_torque = EventTerm(\n')
        p.append('        func=mdp.apply_external_force_torque,\n')
        p.append('        mode="reset",\n')
        p.append('        params={\n')
        p.append('            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),\n')
        p.append(f'            "force_range": {force_range!r},\n')
        p.append(f'            "torque_range": {torque_range!r},\n')
        p.append('        },\n')
        p.append('    )\n\n')

    p.append('    base_external_force_torque_interval = EventTerm(\n')
    p.append('        func=mdp.apply_external_force_torque,\n')
    p.append('        mode="interval",\n')
    p.append(f'        interval_range_s={interval_range_s!r},\n')
    p.append('        params={\n')
    p.append('            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),\n')
    p.append(f'            "force_range": {interval_force!r},\n')
    p.append(f'            "torque_range": {interval_torque!r},\n')
    p.append('        },\n')
    p.append('    )\n\n')

    p.append('    reset_base = EventTerm(\n')
    p.append('        func=mdp.reset_root_state_uniform,\n')
    p.append('        mode="reset",\n')
    p.append('        params={\n')
    p.append(f'            "pose_range": {pose_range!r},\n')
    p.append(f'            "velocity_range": {velocity_range!r},\n')
    p.append('        },\n')
    p.append('    )\n\n')

    p.append('    reset_robot_joints = EventTerm(\n')
    p.append('        func=mdp.reset_joints_by_scale,\n')
    p.append('        mode="reset",\n')
    p.append('        params={\n')
    p.append('            "position_range": (1.0, 1.0),\n')
    p.append(f'            "velocity_range": {joint_vel_range!r},\n')
    p.append('        },\n')
    p.append('    )\n\n')

    if push_enabled:
        p.append('    # interval\n')
        p.append('    push_robot = EventTerm(\n')
        p.append('        func=mdp.push_by_setting_velocity,\n')
        p.append('        mode="interval",\n')
        p.append(f'        interval_range_s={push_interval!r},\n')
        p.append(f'        params={{"velocity_range": {push_velocity!r}}},\n')
        p.append('    )\n')

    return "".join(p)


# ── Terrain scene block builder ────────────────────────────────── #


def _build_terrain_scene_block(terrain_type: str) -> str:
    """Return the `terrain = TerrainImporterCfg(...)` block."""
    if terrain_type == "plane":
        return '''    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )'''
    else:
        return '''    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=COBBLESTONE_ROAD_CFG,
        max_init_terrain_level=COBBLESTONE_ROAD_CFG.num_rows - 1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )'''


# ── Reward block builder ───────────────────────────────────────── #

# Map reward key → (func_name, params_string_or_None)
_REWARD_DEFS = {
    "track_lin_vel_xy": (
        "mdp.track_lin_vel_xy_yaw_frame_exp",
        '"command_name": "base_velocity", "std": math.sqrt(0.25)',
    ),
    "track_ang_vel_z": (
        "mdp.track_ang_vel_z_exp",
        '"command_name": "base_velocity", "std": math.sqrt(0.25)',
    ),
    "alive": ("mdp.is_alive", None),
    "base_linear_velocity": ("mdp.lin_vel_z_l2", None),
    "base_angular_velocity": ("mdp.ang_vel_xy_l2", None),
    "joint_vel": ("mdp.joint_vel_l2", None),
    "joint_acc": ("mdp.joint_acc_l2", None),
    "action_rate_l1": ("mdp.action_rate_l1", None),
    "dof_pos_limits": ("mdp.joint_pos_limits", None),
    "energy": ("mdp.energy", None),
    "joint_deviation_legs": (
        "mdp.joint_deviation_l1",
        '"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])',
    ),
    "joint_deviation_hip_knee": (
        "mdp.joint_deviation_l1",
        '"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_pitch_joint", ".*_knee_joint"])',
    ),
    "flat_orientation_l2": ("mdp.flat_orientation_l2", None),
    "base_height": (
        "mdp.base_height_l2",
        '"target_height": 0.7',
    ),
    "stand_still": (
        "mdp.stand_still_joint_deviation_l1",
        '"asset_cfg": SceneEntityCfg("robot", joint_names=".*"), '
        '"command_name": "base_velocity", "command_threshold": 0.05',
    ),
    "feet_contact_number": (
        "mdp.feet_contact_number",
        '"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"), "period": 0.6',
    ),
    "feet_slide": (
        "mdp.feet_slide",
        '"asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"), '
        '"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*")',
    ),
    "feet_clearance": (
        "mdp.foot_clearance_reward",
        '"std": 0.05, "tanh_mult": 2.0, "target_height": 0.1, '
        '"asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*")',
    ),
    "undesired_contacts": (
        "mdp.undesired_contacts",
        '"threshold": 1, "sensor_cfg": SceneEntityCfg("contact_forces", '
        'body_names=["(?!.*ankle.*).*"])',
    ),
    "joint_mirror": (
        "mdp.joint_mirror",
        '"asset_cfg": SceneEntityCfg("robot"), '
        '"mirror_joints": ['
        '["left_hip_pitch_joint", "right_hip_pitch_joint"], '
        '["left_hip_roll_joint", "right_hip_roll_joint"], '
        '["left_hip_yaw_joint", "right_hip_yaw_joint"], '
        '["left_knee_joint", "right_knee_joint"], '
        '["left_ankle_pitch_joint", "right_ankle_pitch_joint"], '
        '["left_ankle_roll_joint", "right_ankle_roll_joint"]], '
        '"joint_weights": [1.0, 1.0, 1.0, 1.5, 3.0, 1.0]',
    ),
    "termination_penalty": ("mdp.is_terminated", None),
    "feet_air_time": (
        "mdp.feet_air_time_positive_biped",
        '"command_name": "base_velocity", "threshold": 0.4, '
        '"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*")',
    ),
}


# ── Termination block builder ──────────────────────────────────── #

_TERMINATION_DEFS = {
    "time_out": {
        "func": "mdp.time_out",
        "special": "time_out=True",
    },
    "bad_orientation": {
        "func": "mdp.bad_orientation",
        "params_default": '"limit_angle": 0.8',
    },
    "base_height": {
        "func": "mdp.root_height_below_minimum",
        "params_default": '"minimum_height": 0.2',
    },
    "illegal_contact": {
        "func": "mdp.illegal_contact",
        "params_default": (
            '"threshold": 1.0, '
            '"sensor_cfg": SceneEntityCfg("contact_forces", body_names="pelvis")'
        ),
        # Structured defaults for merging with YAML dict overrides
        "params_default_items": [
            ("threshold", 1.0),
            ("sensor_cfg", 'SceneEntityCfg("contact_forces", body_names="pelvis")'),
        ],
    },
}


def _build_terminations_block(terminations_cfg: dict) -> str:
    """Build TerminationsCfg class from YAML config.

    terminations_cfg maps term names to:
      true / "keep" → include with defaults
      null           → skip (not included)
      dict           → include with custom params (overrides defaults)
    """
    lines = [
        '@configclass\n',
        'class TerminationsCfg:\n',
        '    """Termination terms for the MDP."""\n\n',
    ]

    for name, spec in _TERMINATION_DEFS.items():
        if name not in terminations_cfg:
            continue
        val = terminations_cfg[name]
        if val is None:
            continue  # explicitly disabled

        func = spec["func"]

        if spec.get("special"):
            lines.append(f'    {name} = DoneTerm(func={func}, {spec["special"]})\n')
        elif isinstance(val, dict) and val:
            # Custom params from YAML — merge with params_default_items
            default_items = spec.get("params_default_items", [])
            yaml_keys = set(val.keys())
            parts = []
            # Add default items not overridden by YAML
            for dk, dv in default_items:
                if dk not in yaml_keys:
                    if isinstance(dv, str):
                        parts.append(f'"{dk}": {dv}')
                    else:
                        parts.append(f'"{dk}": {dv!r}')
            # Add YAML items (these override defaults)
            for k, v in val.items():
                if isinstance(v, str):
                    parts.append(f'"{k}": {v}')
                else:
                    parts.append(f'"{k}": {v!r}')
            params_str = ", ".join(parts)
            lines.append(f'    {name} = DoneTerm(func={func}, params={{{params_str}}})\n')
        else:
            default_params = spec.get("params_default", "")
            if default_params:
                lines.append(f'    {name} = DoneTerm(func={func}, params={{{default_params}}})\n')
            else:
                lines.append(f'    {name} = DoneTerm(func={func})\n')

    return "".join(lines)


def _build_reward_term(name: str, weight: float) -> str:
    """Return a single RewTerm(...) source line."""
    if name not in _REWARD_DEFS:
        logger.warning("Unknown reward key '%s' — skipping", name)
        return ""

    func, params = _REWARD_DEFS[name]
    try:
        weight_val = float(weight)
    except (TypeError, ValueError):
        return ""
    # Always generate the term even with zero weight — Isaac Lab curriculum
    # may reference reward terms that need to exist regardless of weight.

    if params:
        return f'    {name} = RewTerm(func={func}, weight={weight_val}, params={{{params}}})'
    else:
        return f'    {name} = RewTerm(func={func}, weight={weight_val})'


def _build_rewards_block(rewards: dict) -> str:
    """Build the full RewardsCfg class body."""
    lines = ['@configclass\nclass RewardsCfg:\n    """Reward terms for the MDP."""\n']
    lines.append('    # -- task')
    for rname, weight in rewards.items():
        line = _build_reward_term(rname, weight)
        if line:
            lines.append(line)
    return "\n".join(lines)


# ── Commands block builder ─────────────────────────────────────── #


def _build_commands_block(commands_cfg: dict) -> str:
    """Return source for CommandsCfg class."""
    ranges = commands_cfg.get("ranges", {})
    limit_ranges = commands_cfg.get("limit_ranges", ranges)
    resampling_time_range = tuple(commands_cfg.get("resampling_time_range", (10.0, 10.0)))
    rel_standing_envs = commands_cfg.get("rel_standing_envs", 0.02)
    rel_heading_envs = commands_cfg.get("rel_heading_envs", 1.0)
    heading_command = commands_cfg.get("heading_command", False)
    debug_vis = commands_cfg.get("debug_vis", True)

    def _fmt_range(r: dict, keys: list) -> str:
        parts = []
        for k in keys:
            v = r.get(k, (0.0, 0.0))
            parts.append(f"{k}={v!r}")
        return ", ".join(parts)

    range_keys = ["lin_vel_x", "lin_vel_y", "ang_vel_z"]
    return (
        '@configclass\n'
        'class CommandsCfg:\n'
        '    """Command specifications for the MDP."""\n'
        '\n'
        '    base_velocity = mdp.UniformLevelVelocityCommandCfg(\n'
        '        asset_name="robot",\n'
        f'        resampling_time_range={resampling_time_range!r},\n'
        f'        rel_standing_envs={rel_standing_envs!r},\n'
        f'        rel_heading_envs={rel_heading_envs!r},\n'
        f'        heading_command={heading_command!r},\n'
        f'        debug_vis={debug_vis!r},\n'
        '        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(\n'
        '            ' + _fmt_range(ranges, range_keys) + '\n'
        '        ),\n'
        '        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(\n'
        '            ' + _fmt_range(limit_ranges, range_keys) + '\n'
        '        ),\n'
        '    )'
    )


# ── Full config generator ──────────────────────────────────────── #


def generate_env_config(
    params: dict[str, Any],
    output_path: str | Path,
    project_root: Optional[str | Path] = None,
) -> Path:
    """Generate a complete velocity_env_cfg.py and write it to *output_path*.

    Uses the original velocity_env_cfg.py as a base template and replaces
    only the dynamic sections via regex substitution.

    Parameters
    ----------
    params : dict
        Merged parameters from phase_manager (defaults → phase → sub_phase).
        Expected top-level keys: ``env``, ``rewards``.
    output_path : str or Path
        Where to write the generated Python file.
    project_root : str or Path, optional
        Root of the magiclab_rl_lab project (to find the template).
        If None, attempts to auto-detect.

    Returns
    -------
    Path
        The absolute path of the generated file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Locate the template config
    if project_root is not None:
        template_path = Path(project_root) / _TEMPLATE_CFG_REL
    else:
        # Try relative to this file
        template_path = Path(__file__).resolve().parent.parent.parent / _TEMPLATE_CFG_REL

    if not template_path.exists():
        raise FileNotFoundError(f"Template config not found: {template_path}")

    # Use .orig backup to always read from the pristine template.
    # The active config gets overwritten by _swap_active_config on every
    # sub-phase, so without a backup the regex replacements would operate on
    # an already-modified file and could corrupt the structure (e.g. eat
    # RobotSceneCfg on the second sub-phase).
    orig_path = template_path.parent / (template_path.name + ".orig")
    if orig_path.exists():
        read_path = orig_path
    else:
        # First call: snapshot the current (original) template
        import shutil
        shutil.copy2(str(template_path), str(orig_path))
        read_path = template_path

    template = read_path.read_text(encoding="utf-8")

    env = params.get("env", {})
    rewards = params.get("rewards", {})
    commands = env.get("commands", {
        "ranges": {"lin_vel_x": [-0.5, 1.0], "lin_vel_y": [-0.5, 0.5], "ang_vel_z": [-0.5, 0.5]},
        "limit_ranges": {"lin_vel_x": [-0.5, 1.0], "lin_vel_y": [-0.5, 0.5], "ang_vel_z": [-0.5, 0.5]},
    })
    events = env.get("events", {})

    terrain_type = env.get("terrain_type", "plane")
    terrain_generator = env.get("terrain_generator", None)
    bad_orientation = env.get("bad_orientation_limit", 0.8)
    base_height_min = env.get("base_height_minimum", 0.2)
    decimation = env.get("decimation", 10)
    episode_length_s = env.get("episode_length_s", 20.0)
    sim_dt = env.get("sim_dt", 0.002)
    action_scale = env.get("action_scale", 0.25)
    curriculum_enabled = "True" if terrain_type == "generator" else "False"

    # ── Replacement 1: Terrain generator (COBBLESTONE_ROAD_CFG) ── #
    terrain_gen_block = _build_terrain_generator_block(terrain_generator)
    # Match the entire COBBLESTONE_ROAD_CFG assignment (possibly multi-line)
    # up to the end of the statement.  Use a non-greedy match that stops at
    # the blank line before the next @configclass block.
    template = re.sub(
        r'^COBBLESTONE_ROAD_CFG\s*=\s*.*?(?:\n\n|\n(?=@configclass))',
        terrain_gen_block + '\n\n',
        template,
        count=1,
        flags=re.MULTILINE | re.DOTALL,
    )

    # ── Replacement 2: Terrain scene block ── #
    terrain_scene_block = _build_terrain_scene_block(terrain_type)
    # Match from `terrain = TerrainImporterCfg(` to the closing `)`
    template = re.sub(
        r'    terrain = TerrainImporterCfg\(.*?\n    \)',
        terrain_scene_block,
        template,
        count=1,
        flags=re.DOTALL,
    )

    # ── Replacement 3: CommandsCfg ── #
    events_block = _build_events_block(events)
    template = re.sub(
        r'@configclass\nclass EventCfg:.*?(?=\n\n@configclass\nclass CommandsCfg)',
        events_block + '\n',
        template,
        count=1,
        flags=re.DOTALL,
    )

    commands_block = _build_commands_block(commands)
    template = re.sub(
        r'@configclass\nclass CommandsCfg:.*?(?=\n\n@configclass\nclass ActionsCfg)',
        commands_block + '\n',
        template,
        count=1,
        flags=re.DOTALL,
    )

    # ── Replacement 4: RewardsCfg ── #
    rewards_block = _build_rewards_block(rewards)
    template = re.sub(
        r'@configclass\nclass RewardsCfg:.*?(?=\n\n@configclass\nclass TerminationsCfg)',
        rewards_block + '\n',
        template,
        count=1,
        flags=re.DOTALL,
    )

    # ── Replacement 5: TerminationsCfg ── #
    terminations_cfg = env.get("terminations", None)
    if terminations_cfg is not None:
        # New style: build entire TerminationsCfg from YAML config.
        terminations_block = _build_terminations_block(terminations_cfg)
        template = re.sub(
            r'@configclass\nclass TerminationsCfg:.*?(?=\n\n@configclass\nclass CurriculumCfg)',
            terminations_block + '\n',
            template,
            count=1,
            flags=re.DOTALL,
        )
    else:
        # Legacy: just modify params of existing termination terms.
        template = re.sub(
            r'params=\{"minimum_height": [\d.]+\}',
            f'params={{"minimum_height": {base_height_min}}}',
            template,
        )
        template = re.sub(
            r'params=\{"limit_angle": [\d.]+\}',
            f'params={{"limit_angle": {bad_orientation}}}',
            template,
        )

    # ── Replacement 6: Action scale ── #
    template = re.sub(
        r'scale=[\d.]+, use_default_offset=True',
        f'scale={action_scale}, use_default_offset=True',
        template,
    )

    # ── Replacement 7: Sim params in __post_init__ ── #
    template = re.sub(
        r'self\.decimation = \d+',
        f'self.decimation = {decimation}',
        template,
    )
    template = re.sub(
        r'self\.episode_length_s = [\d.]+',
        f'self.episode_length_s = {episode_length_s}',
        template,
    )
    template = re.sub(
        r'self\.sim\.dt = [\d.]+',
        f'self.sim.dt = {sim_dt}',
        template,
    )

    # ── Replacement 8: CurriculumCfg + __post_init__ curriculum block ── #
    if terrain_type == "plane":
        # No terrain generator → remove terrain_levels curriculum term
        template = re.sub(
            r'    terrain_levels = CurrTerm\(func=mdp\.terrain_levels_vel\)\n',
            '',
            template,
        )
        # Replace the curriculum block in __post_init__ with a simple pass
        # Original has 2 if-blocks checking terrain_levels and setting curriculum
        old_curr_block = (
            r'        # check if terrain levels curriculum is enabled.*?'
            r'self\.scene\.terrain\.terrain_generator\.curriculum = False\n'
        )
        template = re.sub(
            old_curr_block,
            '',
            template,
            flags=re.DOTALL,
        )
        # Fix RobotPlayEnvCfg to handle None terrain_generator
        template = template.replace(
            'self.scene.terrain.terrain_generator.num_rows = 2\n'
            '        self.scene.terrain.terrain_generator.num_cols = 10',
            '# terrain_generator is None in plane mode\n'
            '        if self.scene.terrain.terrain_generator is not None:\n'
            '            self.scene.terrain.terrain_generator.num_rows = 2\n'
            '            self.scene.terrain.terrain_generator.num_cols = 10',
        )
    else:
        template = re.sub(
            r'self\.scene\.terrain\.terrain_generator\.curriculum = (True|False)',
            f'self.scene.terrain.terrain_generator.curriculum = {curriculum_enabled}',
            template,
        )

    output_path.write_text(template, encoding="utf-8")
    logger.info("Generated env config: %s", output_path)
    return output_path.resolve()
