"""Z1 throwing environment configuration — parameterized hand design.

换手部 URDF 时只需修改 HAND_CONFIG，训练逻辑不变。
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from magiclab_rl_lab.assets.robots.magiclab import MAGICLAB_Z1_23DOF_CFG as ROBOT_CFG
from magiclab_rl_lab.tasks.throwing import mdp


# ---------------------------------------------------------------------------
# Hand configuration — swap here when Z1 dexterous hand URDF arrives
# ---------------------------------------------------------------------------
HAND_CONFIG = {
    "hand_type": "fixed_palm",           # "fixed_palm" or "dexterous"
    "hand_joint_names": [],              # fill when dexterous
    "hand_dof": 0,                       # fill when dexterous
    "ee_link": "left_hand_palm_link",    # end-effector link
    "ball_support_link": "left_hand_palm_link",
}

# Joint groups
LEG_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]

ARM_JOINTS = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_yaw_joint",
]

WAIST_JOINTS = ["waist_yaw_joint"]

# Dynamic action joints based on hand type
if HAND_CONFIG["hand_type"] == "fixed_palm":
    ACTION_JOINTS = ARM_JOINTS + WAIST_JOINTS  # 11 DOF
elif HAND_CONFIG["hand_type"] == "dexterous":
    ACTION_JOINTS = ARM_JOINTS + HAND_CONFIG["hand_joint_names"] + WAIST_JOINTS
else:
    ACTION_JOINTS = ARM_JOINTS + WAIST_JOINTS

ALL_JOINTS = LEG_JOINTS + ARM_JOINTS + WAIST_JOINTS

# EE body name for observations
EE_BODY_NAME = HAND_CONFIG["ee_link"]


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------
@configclass
class ThrowingSceneCfg(InteractiveSceneCfg):
    """Scene with robot, ball, target basket, and ground."""

    # Ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    # Robot
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Ball (rigid object)
    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim_utils.SphereCfg(
            radius=0.06,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.15),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8,
                dynamic_friction=0.6,
                restitution=0.7,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.5, 0.0),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.0),
        ),
    )

    # Target basket (static visual + collision for detection)
    basket = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Basket",
        spawn=sim_utils.CylinderCfg(
            radius=0.15,
            height=0.05,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0),
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False,  # trigger only, no physics
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(2.0, 0.0, 1.5),
        ),
    )

    # Lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=None,
        ),
    )


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------
@configclass
class EventCfg:
    """Configuration for events."""

    # Reset robot base
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-0.1, 0.1)},
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        },
    )

    # Reset joints to default
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------
@configclass
class ActionsCfg:
    """Action specifications — arm + waist joints."""

    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=ACTION_JOINTS,
        scale=0.5,
        use_default_offset=True,
        preserve_order=True,
    )


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------
@configclass
class ObservationsCfg:
    """Observation specifications for the throwing task."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy (no privileged info)."""

        # Proprioception
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ALL_JOINTS, preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ALL_JOINTS, preserve_order=True)},
            scale=0.1,
            noise=Unoise(n_min=-0.5, n_max=0.5),
        )
        last_action = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0))

        # Ball state
        ball_pos = ObsTerm(
            func=mdp.ball_position,
            params={"ball_cfg": SceneEntityCfg("ball")},
        )
        ball_vel = ObsTerm(
            func=mdp.ball_velocity,
            params={"ball_cfg": SceneEntityCfg("ball")},
            scale=0.1,
        )
        ball_on_palm = ObsTerm(
            func=mdp.ball_on_palm_flag,
            params={"ball_cfg": SceneEntityCfg("ball"), "asset_cfg": SceneEntityCfg("robot", body_names=EE_BODY_NAME)},
        )

        # Target
        target_pos = ObsTerm(
            func=mdp.target_position,
            params={"target_cfg": SceneEntityCfg("basket")},
        )

        # EE state
        ee_pos = ObsTerm(
            func=mdp.ee_position,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=EE_BODY_NAME)},
        )
        ee_vel = ObsTerm(
            func=mdp.ee_velocity,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=EE_BODY_NAME)},
            scale=0.1,
        )

        def __post_init__(self):
            self.history_length = 3
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged observations for critic."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ALL_JOINTS, preserve_order=True)},
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ALL_JOINTS, preserve_order=True)},
            scale=0.1,
        )
        last_action = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0))

        ball_pos = ObsTerm(
            func=mdp.ball_position,
            params={"ball_cfg": SceneEntityCfg("ball")},
        )
        ball_vel = ObsTerm(
            func=mdp.ball_velocity,
            params={"ball_cfg": SceneEntityCfg("ball")},
        )
        ball_on_palm = ObsTerm(
            func=mdp.ball_on_palm_flag,
            params={"ball_cfg": SceneEntityCfg("ball"), "asset_cfg": SceneEntityCfg("robot", body_names=EE_BODY_NAME)},
        )
        target_pos = ObsTerm(
            func=mdp.target_position,
            params={"target_cfg": SceneEntityCfg("basket")},
        )
        ee_pos = ObsTerm(
            func=mdp.ee_position,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=EE_BODY_NAME)},
        )
        ee_vel = ObsTerm(
            func=mdp.ee_velocity,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=EE_BODY_NAME)},
        )

        def __post_init__(self):
            self.history_length = 3

    critic: CriticCfg = CriticCfg()


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------
@configclass
class RewardsCfg:
    """Reward terms for the throwing task."""

    # -- Task rewards
    ball_on_palm = RewTerm(
        func=mdp.ball_on_palm,
        weight=2.0,
        params={
            "ball_cfg": SceneEntityCfg("ball"),
            "ee_body_name": EE_BODY_NAME,
            "threshold": 0.08,
        },
    )
    ball_distance = RewTerm(
        func=mdp.ball_distance_to_target,
        weight=1.0,
        params={
            "ball_cfg": SceneEntityCfg("ball"),
            "target_cfg": SceneEntityCfg("basket"),
        },
    )
    ball_release_vel = RewTerm(
        func=mdp.ball_release_velocity_reward,
        weight=3.0,
        params={
            "ball_cfg": SceneEntityCfg("ball"),
            "target_cfg": SceneEntityCfg("basket"),
            "ee_body_name": EE_BODY_NAME,
        },
    )
    ball_in_basket = RewTerm(
        func=mdp.ball_in_basket,
        weight=10.0,
        params={
            "ball_cfg": SceneEntityCfg("ball"),
            "target_cfg": SceneEntityCfg("basket"),
        },
    )

    # -- Penalties
    ball_fall = RewTerm(
        func=mdp.ball_fall_penalty,
        weight=-2.0,
        params={"ball_cfg": SceneEntityCfg("ball")},
    )
    arm_energy = RewTerm(
        func=mdp.arm_energy_penalty,
        weight=-1e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS, preserve_order=True)},
    )
    joint_limits = RewTerm(
        func=mdp.joint_limit_penalty,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTION_JOINTS, preserve_order=True)},
    )
    action_rate = RewTerm(func=mdp.action_rate_l1, weight=-0.01)
    base_height = RewTerm(func=mdp.base_height_l2, weight=-5.0, params={"target_height": 0.69})
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTION_JOINTS, preserve_order=True)},
    )


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------
@configclass
class TerminationsCfg:
    """Termination terms for the throwing task."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.0})


# ---------------------------------------------------------------------------
# Main env config
# ---------------------------------------------------------------------------
@configclass
class ThrowingEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Z1 throwing environment."""

    scene: ThrowingSceneCfg = ThrowingSceneCfg(num_envs=4096, env_spacing=5.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 8
        self.episode_length_s = 5.0  # short episodes for throwing
        self.sim.dt = 0.004
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material


@configclass
class ThrowingPlayEnvCfg(ThrowingEnvCfg):
    """Play/test configuration with fewer envs."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
