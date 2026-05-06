"""
Go2 locomotion training script  —  IsaacLab + RSL-RL PPO.

Run (headless, 4096 envs):
    python go2_rl_train.py --headless --num_envs 4096

Run with GUI (fewer envs to reduce VRAM):
    python go2_rl_train.py --num_envs 512

Resume from checkpoint:
    python go2_rl_train.py --headless --num_envs 4096 --resume logs/rsl_rl/go2/.../model_500.pt

NOTE: SimulationApp must be created before any omni/isaac imports.
      Do not move the AppLauncher block below other imports.
"""
from __future__ import annotations
import argparse
import os
from datetime import datetime

# ── 1. AppLauncher FIRST ─────────────────────────────────────────────────────
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train Go2 locomotion with RSL-RL PPO")
parser.add_argument("--num_envs", type=int, default=4096, help="Parallel environments")
parser.add_argument("--max_iterations", type=int, default=1500, help="PPO update iterations")
parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
parser.add_argument("--device", type=str, default="cuda:0")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 2. Remaining imports after SimulationApp is live ─────────────────────────
import torch  # noqa: E402
from go2_rl_env import Go2LocomotionEnv, Go2LocomotionEnvCfg  # noqa: E402

# Prefer the isaaclab_rl wrapper (ships with IsaacLab); fall back to a thin shim.
try:
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
    _USE_ISL_WRAPPER = True
except ImportError:
    _USE_ISL_WRAPPER = False

from rsl_rl.runners import OnPolicyRunner  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# RSL-RL PPO configuration dict
# ─────────────────────────────────────────────────────────────────────────────
# Adjust actor/critic hidden dims, learning rate, and entropy coef to taste.
# These defaults work well for flat-terrain Go2 locomotion.

def build_runner_cfg(args: argparse.Namespace) -> dict:
    return {
        "seed": 42,
        "device": args.device,
        "num_steps_per_env": 24,          # rollout horizon per env before PPO update
        "max_iterations": args.max_iterations,
        "save_interval": 100,             # save checkpoint every N iterations
        "experiment_name": "go2_locomotion",
        "run_name": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "logger": "tensorboard",
        "empirical_normalization": False,
        "policy": {
            "class_name": "ActorCritic",
            "init_noise_std": 1.0,
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "activation": "elu",
        },
        "algorithm": {
            "class_name": "PPO",
            "value_loss_coef": 1.0,
            "use_clipping": True,
            "clip_param": 0.2,
            "entropy_coef": 0.008,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 1e-3,
            "schedule": "adaptive",   # auto-adjust LR to hit desired_kl
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Minimal RSL-RL VecEnv shim (used only when isaaclab_rl is not installed)
# ─────────────────────────────────────────────────────────────────────────────

class _MinimalRslWrapper:
    """Wraps Go2LocomotionEnv to match rsl_rl.env.VecEnv interface."""

    def __init__(self, env: Go2LocomotionEnv):
        self.env = env
        self.num_envs: int = env.num_envs
        self.num_obs: int = env.cfg.observation_space
        self.num_privileged_obs = None   # no asymmetric actor-critic
        self.num_actions: int = env.cfg.action_space
        self.max_episode_length: int = env.max_episode_length
        self.device: str = env.device

    # RSL-RL calls reset() once at the start
    def reset(self):
        obs_dict, _ = self.env.reset()
        return obs_dict["policy"], None

    # RSL-RL calls step() every rollout step
    def step(self, actions: torch.Tensor):
        obs_dict, rew, terminated, truncated, info = self.env.step(actions)
        dones = terminated | truncated
        return obs_dict["policy"], rew, dones, info

    def get_observations(self):
        obs_dict = self.env.obs_buf
        return obs_dict["policy"], {}

    def close(self):
        self.env.close()

    # RSL-RL reads/writes episode_length_buf to track resets
    @property
    def episode_length_buf(self):
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.env.episode_length_buf = value


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Build env
    env_cfg = Go2LocomotionEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    env = Go2LocomotionEnv(cfg=env_cfg, render_mode="rgb_array" if args_cli.headless else None)

    # Debug: print resolved body/joint names so you can verify foot patterns and actuators.
    # If body names don't contain "_foot", update contact_sensor_cfg prim_path in go2_rl_env.py.
    print("[go2_rl_train] Joint names :", env.robot.data.joint_names)
    print("[go2_rl_train] Body names  :", env.robot.data.body_names)

    # Wrap for RSL-RL
    if _USE_ISL_WRAPPER:
        wrapped_env = RslRlVecEnvWrapper(env)
    else:
        print("[go2_rl_train] isaaclab_rl not found — using minimal RSL-RL shim.")
        wrapped_env = _MinimalRslWrapper(env)

    # Log directory
    log_dir = os.path.join(
        "logs", "rsl_rl", "go2_locomotion", datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    os.makedirs(log_dir, exist_ok=True)
    print(f"[go2_rl_train] Logging to: {log_dir}")

    runner_cfg = build_runner_cfg(args_cli)
    runner = OnPolicyRunner(wrapped_env, runner_cfg, log_dir=log_dir, device=args_cli.device)

    if args_cli.resume:
        print(f"[go2_rl_train] Resuming from: {args_cli.resume}")
        runner.load(args_cli.resume)

    runner.learn(
        num_learning_iterations=args_cli.max_iterations,
        init_at_random_ep_len=True,
    )

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
