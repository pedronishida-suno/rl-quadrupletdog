"""
Go2 locomotion + fall-recovery RL environment.
Target: Isaac Sim 5.1 / IsaacLab / RSL-RL (DirectRLEnv pattern).

Observation  : 48-dim  [lin_vel(3) ang_vel(3) gravity(3) cmd(3) dq(12) dqdt(12) prev_act(12)]
Action       : 12-dim  joint position deltas from stand pose, scaled by action_scale
Termination  : base height < 0.08 m  OR  time limit (grace period for fall-spawns)
Recovery     : 30 % of resets spawn the robot on its side; orientation reward drives righting
"""
from __future__ import annotations
import math
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

# ─────────────────────────────────────────────────────────────────────────────
# Robot asset
# ─────────────────────────────────────────────────────────────────────────────

_GO2_USD = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
    "/Assets/Isaac/5.1/Isaac/Robots/Unitree/Go2/go2.usd"
)

# Default stand pose (matches go2_start.py POSE_STAND; order: FL FR RL RR × hip/thigh/calf)
_STAND_POSE: dict[str, float] = {
    "FL_hip_joint":   0.1,  "FR_hip_joint":  -0.1,
    "RL_hip_joint":   0.1,  "RR_hip_joint":  -0.1,
    "FL_thigh_joint": 0.8,  "FR_thigh_joint": 0.8,
    "RL_thigh_joint": 0.8,  "RR_thigh_joint": 0.8,
    "FL_calf_joint": -1.5,  "FR_calf_joint": -1.5,
    "RL_calf_joint": -1.5,  "RR_calf_joint": -1.5,
}

GO2_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=_GO2_USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.42),
        joint_pos=_STAND_POSE,
        joint_vel={".*": 0.0},
    ),
    actuators={
        # Soft PD — appropriate for RL (go2_start.py uses very stiff position control)
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=33.5,
            velocity_limit=21.0,
            stiffness=25.0,   # Nm/rad
            damping=0.5,      # Nm/(rad/s)
        ),
    },
    soft_joint_pos_limit_factor=0.9,
)


# ─────────────────────────────────────────────────────────────────────────────
# Environment configuration
# ─────────────────────────────────────────────────────────────────────────────

@configclass
class Go2LocomotionEnvCfg(DirectRLEnvCfg):
    # simulation: physics at 200 Hz, policy runs at 50 Hz (decimation = 4)
    sim: SimulationCfg = SimulationCfg(dt=1.0 / 200.0, render_interval=4)

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=2.5, replicate_physics=True
    )

    robot_cfg: ArticulationCfg = GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # contact sensors scoped to feet only
    contact_sensor_cfg: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_foot",
        history_length=3,
        track_air_time=True,
    )

    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    decimation: int = 4
    episode_length_s: float = 20.0

    # RL spaces
    action_space: int = 12
    observation_space: int = 48
    state_space: int = 0

    # network output [-1,1] → joint delta [rad]
    action_scale: float = 0.5

    # velocity command sampling ranges [m/s or rad/s]
    cmd_vx_range: tuple[float, float] = (-1.0, 1.0)
    cmd_vy_range: tuple[float, float] = (-0.5, 0.5)
    cmd_yaw_range: tuple[float, float] = (-1.0, 1.0)
    cmd_resample_time_s: float = 10.0

    # fall-recovery curriculum: fraction of resets that spawn on side
    fall_spawn_prob: float = 0.30

    # episode terminates if base height drops below this.
    # Go2 lying on its side has COM ~0.13–0.15 m; 0.08 m = truly stuck on ground.
    # Set higher (e.g. 0.18) if you want the walker only, lower to allow recovery attempts.
    termination_height: float = 0.08  # [m]

    # policy steps of grace after a fall-spawn before termination is checked.
    # Gives the agent time to attempt righting without dying on the first physics settle.
    fall_grace_steps: int = 80

    # reward weights  (tune these; negative = penalty)
    rew_lin_vel_tracking: float = 1.5
    rew_ang_vel_tracking: float = 0.75
    rew_alive: float = 0.5          # constant per step — encourages surviving to recover
    rew_orientation: float = -2.0   # penalise tilt; drives righting behaviour
    rew_base_height: float = -0.5   # penalise deviation from 0.42 m
    rew_joint_torques: float = -2e-5
    rew_action_rate: float = -0.01
    rew_feet_air_time: float = 0.5  # reward regular gait footfalls


# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────

class Go2LocomotionEnv(DirectRLEnv):
    cfg: Go2LocomotionEnvCfg

    def __init__(self, cfg: Go2LocomotionEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # joint indices resolved after scene is built
        self._joint_ids, _ = self.robot.find_joints(".*")
        self._n_dof = len(self._joint_ids)

        # [N, 12] default stand positions
        self._default_joint_pos = self.robot.data.default_joint_pos.clone()

        # velocity commands  [N, 3] = (vx, vy, yaw_rate)
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        # action history for rate penalty and as observation
        self._prev_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._actions = torch.zeros_like(self._prev_actions)

        # how many policy steps since last command resample
        self._cmd_step_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._cmd_resample_steps = int(
            self.cfg.cmd_resample_time_s / (self.cfg.decimation * self.cfg.sim.dt)
        )

        # per-foot air-time accumulator [N, 4]
        self._feet_air_time = torch.zeros(self.num_envs, 4, device=self.device)

        # grace-period counter: envs with value > 0 skip height-termination.
        # Prevents fall-spawned episodes from dying on the first physics settle.
        self._grace_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    # ── scene ────────────────────────────────────────────────────────────────

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot

        self.contact_sensor = ContactSensor(self.cfg.contact_sensor_cfg)
        self.scene.sensors["contact_sensor"] = self.contact_sensor

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ── action application ───────────────────────────────────────────────────

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._joint_targets = self._default_joint_pos + self.cfg.action_scale * self._actions

    def _apply_action(self):
        self.robot.set_joint_position_target(self._joint_targets, joint_ids=self._joint_ids)

    # ── observations ─────────────────────────────────────────────────────────

    def _get_observations(self) -> dict:
        self._cmd_step_counter += 1
        resample_ids = (self._cmd_step_counter >= self._cmd_resample_steps).nonzero(
            as_tuple=False
        ).squeeze(1)
        if len(resample_ids) > 0:
            self._resample_commands(resample_ids)

        obs = torch.cat(
            [
                self.robot.data.root_lin_vel_b,                          # [N,3]
                self.robot.data.root_ang_vel_b,                          # [N,3]
                self.robot.data.projected_gravity_b,                     # [N,3]
                self._commands,                                           # [N,3]
                self.robot.data.joint_pos - self._default_joint_pos,     # [N,12]
                self.robot.data.joint_vel,                               # [N,12]
                self._prev_actions,                                       # [N,12]
            ],
            dim=-1,
        )  # total: 48
        return {"policy": obs}

    # ── rewards ──────────────────────────────────────────────────────────────

    def _get_rewards(self) -> torch.Tensor:
        lin_vel_b = self.robot.data.root_lin_vel_b       # [N,3]
        ang_vel_b = self.robot.data.root_ang_vel_b       # [N,3]
        gravity_b = self.robot.data.projected_gravity_b  # [N,3]

        # velocity tracking (exponential kernel keeps gradient near zero-error)
        lin_err = torch.sum((self._commands[:, :2] - lin_vel_b[:, :2]) ** 2, dim=1)
        r_lin = torch.exp(-lin_err / 0.25)
        ang_err = (self._commands[:, 2] - ang_vel_b[:, 2]) ** 2
        r_ang = torch.exp(-ang_err / 0.25)

        # orientation: gravity_b = [0,0,-1] when perfectly upright; lateral components penalised
        r_orient = torch.sum(gravity_b[:, :2] ** 2, dim=1)

        # base height deviation from nominal 0.42 m
        r_height = (self.robot.data.root_pos_w[:, 2] - 0.42) ** 2

        # mechanical cost
        r_torque = torch.sum(self.robot.data.applied_torque ** 2, dim=1)
        r_action_rate = torch.sum((self._actions - self._prev_actions) ** 2, dim=1)

        # gait quality: reward feet that spend >0.5 s in air before touching down
        contact_z = self.contact_sensor.data.net_forces_w[:, :, 2]   # [N, 4]
        in_contact = contact_z > 1.0                                   # [N, 4]
        first_contact = in_contact & (self._feet_air_time > 0.0)
        r_air = torch.sum(
            torch.clamp(self._feet_air_time - 0.5, min=0.0) * first_contact.float(), dim=1
        )
        dt_step = self.cfg.decimation * self.cfg.sim.dt
        self._feet_air_time += dt_step
        self._feet_air_time *= ~in_contact   # reset to 0 when foot touches ground

        # update prev actions after using them for the penalty
        self._prev_actions = self._actions.clone()

        return (
            self.cfg.rew_lin_vel_tracking * r_lin
            + self.cfg.rew_ang_vel_tracking * r_ang
            + self.cfg.rew_orientation * r_orient
            + self.cfg.rew_base_height * r_height
            + self.cfg.rew_joint_torques * r_torque
            + self.cfg.rew_action_rate * r_action_rate
            + self.cfg.rew_feet_air_time * r_air
            + self.cfg.rew_alive
        )

    # ── terminations ─────────────────────────────────────────────────────────

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # only terminate when height is truly unrecoverable; allows righting attempts.
        # grace_buf suppresses termination for fall-spawned envs for the first N steps.
        height_too_low = self.robot.data.root_pos_w[:, 2] < self.cfg.termination_height
        fallen = height_too_low & (self._grace_buf == 0)
        self._grace_buf = torch.clamp(self._grace_buf - 1, min=0)
        return fallen, time_out

    # ── resets ───────────────────────────────────────────────────────────────

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == 0:
            return
        super()._reset_idx(env_ids)

        n = len(env_ids)

        # reset joints to stand pose
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # reset base pose (translate to each env's world origin)
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]

        # ── fall-spawn curriculum ────────────────────────────────────────────
        # Spawn a fraction of envs lying on their side so the agent learns to recover.
        fall_mask = torch.rand(n, device=self.device) < self.cfg.fall_spawn_prob
        if fall_mask.any():
            n_fall = int(fall_mask.sum().item())
            # random roll ±(72°–90°) — clearly on side but reachable from flat ground
            sign = torch.where(
                torch.rand(n_fall, device=self.device) > 0.5,
                torch.ones(n_fall, device=self.device),
                -torch.ones(n_fall, device=self.device),
            )
            roll = sign * (math.pi * 0.40 + torch.rand(n_fall, device=self.device) * math.pi * 0.10)
            half = roll * 0.5
            q_fall = torch.zeros(n_fall, 4, device=self.device)
            q_fall[:, 0] = torch.cos(half)   # w
            q_fall[:, 1] = torch.sin(half)   # x  → rotation around body-forward = roll
            root_state[fall_mask, 3:7] = q_fall
            root_state[fall_mask, 2] = 0.28  # lower COM when lying on side
            # grant grace period so physics settle doesn't immediately trigger termination
            self._grace_buf[env_ids[fall_mask]] = self.cfg.fall_grace_steps

        self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)

        # reset RL buffers
        self._commands[env_ids] = 0.0
        self._resample_commands(env_ids)
        self._prev_actions[env_ids] = 0.0
        self._actions[env_ids] = 0.0
        self._cmd_step_counter[env_ids] = 0
        self._feet_air_time[env_ids] = 0.0
        # grace_buf reset for non-fall envs (fall envs already set theirs above)
        non_fall_ids = env_ids[~fall_mask]
        if len(non_fall_ids) > 0:
            self._grace_buf[non_fall_ids] = 0

    # ── helpers ──────────────────────────────────────────────────────────────

    def _resample_commands(self, env_ids: torch.Tensor):
        n = len(env_ids)
        vx_lo, vx_hi = self.cfg.cmd_vx_range
        vy_lo, vy_hi = self.cfg.cmd_vy_range
        yaw_lo, yaw_hi = self.cfg.cmd_yaw_range
        self._commands[env_ids, 0] = torch.empty(n, device=self.device).uniform_(vx_lo, vx_hi)
        self._commands[env_ids, 1] = torch.empty(n, device=self.device).uniform_(vy_lo, vy_hi)
        self._commands[env_ids, 2] = torch.empty(n, device=self.device).uniform_(yaw_lo, yaw_hi)
        self._cmd_step_counter[env_ids] = 0
