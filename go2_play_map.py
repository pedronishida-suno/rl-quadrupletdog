"""
Go2 — playback da política treinada dentro de um cenário 3D.

Carrega um checkpoint PPO (treinado por go2_rl_train.py) e roda a política num
ambiente 3D (cenário built-in do Isaac OU um USD próprio — ex.: o mapa da UFU /
Santa Mônica depois de convertido para .usd). O Go2 é dirigido por teclado, que
injeta o comando de velocidade (vx, vy, yaw) que a política espera.

A observação, o escalonamento de ação e a pose padrão são REUSADOS de
go2_rl_env.py — então o que a política vê aqui é idêntico ao do treino.

Exemplos:
    # cenário built-in Warehouse, política dirigida por teclado
    isaaclab.sh -p go2_play_map.py --checkpoint logs/rsl_rl/go2_locomotion/<run>/model_1500.pt

    # outro cenário built-in
    isaaclab.sh -p go2_play_map.py --checkpoint <ckpt>.pt --map office

    # mapa próprio (o .skp da UFU/Santa Mônica depois de virar .usd)
    isaaclab.sh -p go2_play_map.py --checkpoint <ckpt>.pt \
        --map /home/pedro/maps/santa_monica.usd --map_scale 0.0254

Teclado (com a janela do Isaac em foco):
    W / S         frente / trás       (vx)
    A / D         esquerda / direita  (vy)
    Q / E         girar esq. / dir.   (yaw)
    espaço        parar (zera comando)

NOTA: SimulationApp/AppLauncher deve ser criado ANTES de qualquer import omni/isaac.
"""
from __future__ import annotations
import argparse

# ── 1. AppLauncher PRIMEIRO ──────────────────────────────────────────────────
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Playback da política Go2 num cenário 3D")
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Caminho do checkpoint .pt treinado por go2_rl_train.py")
parser.add_argument("--map", type=str, default="warehouse",
                    help="Chave built-in (warehouse|warehouse_shelves|office|hospital|flat) "
                         "OU caminho/URL para um .usd próprio")
parser.add_argument("--map_scale", type=float, default=1.0,
                    help="Fator de escala aplicado ao USD do mapa (use ~0.0254 se o "
                         "mapa estiver em polegadas, comum em exports de SketchUp)")
parser.add_argument("--map_z", type=float, default=0.0,
                    help="Deslocamento vertical do mapa (m), p/ alinhar o piso em z=0")
parser.add_argument("--num_envs", type=int, default=1, help="Nº de robôs (demo: 1)")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--isaaclab_flat", action="store_true",
                    help="Modo compatível com o checkpoint OFICIAL Isaac-Velocity-Flat-Unitree-Go2-v0: "
                         "rede [128,128,128], action_scale 0.25, coxa traseira 1.0, atuador DCMotor, "
                         "e o go2.usd do IsaacLab (mesma ordem de juntas do treino).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 2. Imports após SimulationApp estar vivo ─────────────────────────────────
import copy  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import DCMotorCfg  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.sensors import ContactSensor  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from go2_rl_env import GO2_CFG, Go2LocomotionEnv, Go2LocomotionEnvCfg  # noqa: E402
from go2_rl_train import build_runner_cfg  # noqa: E402

# Caminho-raiz dos assets do Isaac (Nucleus / S3) e do IsaacLab.
try:
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR  # noqa: E402
except Exception:  # fallback p/ versões antigas
    from isaacsim.storage.native import get_assets_root_path  # noqa: E402
    _root = get_assets_root_path() or ""
    ISAAC_NUCLEUS_DIR = _root + "/Isaac"
    ISAACLAB_NUCLEUS_DIR = _root + "/Isaac/IsaacLab"


# ─────────────────────────────────────────────────────────────────────────────
# Config compatível com o checkpoint OFICIAL Isaac-Velocity-Flat-Unitree-Go2-v0.
# Casa pose padrão, atuador, action_scale, arquitetura de rede E o próprio USD
# (mesma ordem de juntas com que a política foi treinada).
# ─────────────────────────────────────────────────────────────────────────────

# Pose padrão do IsaacLab: coxas TRASEIRAS = 1.0 (as nossas são 0.8).
_ISL_STAND_POSE: dict[str, float] = {
    "FL_hip_joint":   0.1,  "FR_hip_joint":  -0.1,
    "RL_hip_joint":   0.1,  "RR_hip_joint":  -0.1,
    "FL_thigh_joint": 0.8,  "FR_thigh_joint": 0.8,
    "RL_thigh_joint": 1.0,  "RR_thigh_joint": 1.0,
    "FL_calf_joint": -1.5,  "FR_calf_joint": -1.5,
    "RL_calf_joint": -1.5,  "RR_calf_joint": -1.5,
}

# Rede do checkpoint flat oficial (rough usa [512,256,128] + height_scan).
ISL_FLAT_HIDDEN_DIMS = [128, 128, 128]


def build_isaaclab_flat_robot_cfg():
    """GO2_CFG ajustado p/ casar exatamente o robô do checkpoint flat oficial."""
    cfg = copy.deepcopy(GO2_CFG)
    # mesmo USD do treino → mesma ordem de DOFs
    cfg.spawn.usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/Go2/go2.usd"
    cfg.init_state.joint_pos = _ISL_STAND_POSE
    # atuador DCMotor do IsaacLab (substitui o ImplicitActuator do nosso treino)
    cfg.actuators = {
        "base_legs": DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=23.5,
            saturation_effort=23.5,
            velocity_limit=30.0,
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
        ),
    }
    return cfg

# Cenários 3D prontos que acompanham o Isaac Sim. "flat" = só o ground plane.
BUILTIN_MAPS = {
    "warehouse":        f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd",
    "warehouse_shelves": f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd",
    "office":           f"{ISAAC_NUCLEUS_DIR}/Environments/Office/office.usd",
    "hospital":         f"{ISAAC_NUCLEUS_DIR}/Environments/Hospital/hospital.usd",
    "flat":             "",
}


# ─────────────────────────────────────────────────────────────────────────────
# Config de playback: 1 robô, sem fall-spawn, comando dirigido por teclado,
# e um USD de cenário opcional carregado como geometria estática global.
# ─────────────────────────────────────────────────────────────────────────────
@configclass
class Go2PlayEnvCfg(Go2LocomotionEnvCfg):
    map_usd_path: str = ""
    map_scale: tuple = (1.0, 1.0, 1.0)
    map_translation: tuple = (0.0, 0.0, 0.0)


class Go2PlayEnv(Go2LocomotionEnv):
    """Go2LocomotionEnv + cenário 3D estático. Observação/ação idênticas ao treino."""

    cfg: Go2PlayEnvCfg

    def _setup_scene(self):
        # robô + sensor de contato (igual ao env de treino)
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot

        self.contact_sensor = ContactSensor(self.cfg.contact_sensor_cfg)
        self.scene.sensors["contact_sensor"] = self.contact_sensor

        # piso plano com colisão (a política foi treinada sobre ele)
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # mapa 3D como prim estático global (visual + colisão)
        global_paths = [self.cfg.terrain.prim_path]
        if self.cfg.map_usd_path:
            map_cfg = sim_utils.UsdFileCfg(
                usd_path=self.cfg.map_usd_path,
                scale=self.cfg.map_scale,
            )
            map_cfg.func("/World/Map", map_cfg, translation=self.cfg.map_translation)
            global_paths.append("/World/Map")
            print(f"[go2_play] Mapa carregado: {self.cfg.map_usd_path} "
                  f"(escala={self.cfg.map_scale}, z={self.cfg.map_translation[2]})")

        self.scene.clone_environments(copy_from_source=False)
        # não filtra colisões contra o piso nem contra o mapa → robô colide com ambos
        self.scene.filter_collisions(global_prim_paths=global_paths)

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)


# ─────────────────────────────────────────────────────────────────────────────
# Teclado → comando de velocidade no corpo (vx, vy, yaw)
# ─────────────────────────────────────────────────────────────────────────────
class KeyboardCommand:
    """Lê WASD/QE e devolve (vx, vy, yaw) dentro das faixas de comando do treino."""

    def __init__(self, vx_max=1.0, vy_max=0.5, yaw_max=1.0):
        self.vx_max, self.vy_max, self.yaw_max = vx_max, vy_max, yaw_max
        self._keys: dict = {}
        self._ok = False
        try:
            import carb.input
            import omni.appwindow
            self._carb = carb.input
            self._input = carb.input.acquire_input_interface()
            self._kbd = omni.appwindow.get_default_app_window().get_keyboard()
            self._sub = self._input.subscribe_to_keyboard_events(self._kbd, self._on_key)
            self._ok = True
        except Exception as exc:  # headless ou sem janela
            print(f"[go2_play] Teclado indisponível ({exc}); usando comando fixo p/ frente.")

    def _on_key(self, event, *a, **k):
        try:
            kk = event.input
            if event.type == self._carb.KeyboardEventType.KEY_PRESS:
                self._keys[kk] = True
            elif event.type == self._carb.KeyboardEventType.KEY_RELEASE:
                self._keys[kk] = False
        except Exception:
            pass
        return True

    def get(self) -> tuple[float, float, float]:
        if not self._ok:
            return (0.5, 0.0, 0.0)  # anda devagar p/ frente em headless
        K = self._carb.KeyboardInput
        down = lambda key: self._keys.get(key, False)
        vx = (down(K.W) - down(K.S)) * self.vx_max
        vy = (down(K.A) - down(K.D)) * self.vy_max
        yaw = (down(K.Q) - down(K.E)) * self.yaw_max
        if down(K.SPACE):
            vx = vy = yaw = 0.0
        return (float(vx), float(vy), float(yaw))


# ─────────────────────────────────────────────────────────────────────────────
# Carregamento da política (rsl_rl ActorCritic + state_dict do checkpoint)
# ─────────────────────────────────────────────────────────────────────────────
def load_policy(checkpoint_path: str, num_obs: int, num_actions: int, device: str,
                hidden_dims: list[int] | None = None):
    from rsl_rl.modules import ActorCritic

    policy_cfg = dict(build_runner_cfg(args_cli)["policy"])
    policy_cfg.pop("class_name", None)
    # sobrescreve a arquitetura quando o checkpoint pede dims diferentes
    if hidden_dims is not None:
        policy_cfg["actor_hidden_dims"] = list(hidden_dims)
        policy_cfg["critic_hidden_dims"] = list(hidden_dims)

    actor_critic = ActorCritic(num_obs, num_obs, num_actions, **policy_cfg).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    actor_critic.load_state_dict(state)
    actor_critic.eval()
    print(f"[go2_play] Política carregada de {checkpoint_path}")
    return lambda obs: actor_critic.act_inference(obs)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # resolve o caminho do mapa (chave built-in ou caminho/URL)
    map_path = BUILTIN_MAPS.get(args_cli.map, args_cli.map)

    cfg = Go2PlayEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.fall_spawn_prob = 0.0               # demo: nunca nasce de lado
    cfg.cmd_resample_time_s = 1e9           # nunca reamostra sozinho — teclado manda
    cfg.map_usd_path = map_path
    cfg.map_scale = (args_cli.map_scale,) * 3
    cfg.map_translation = (0.0, 0.0, args_cli.map_z)

    hidden_dims = None
    if args_cli.isaaclab_flat:
        # casa exatamente o robô/escala/rede do checkpoint flat oficial
        cfg.robot_cfg = build_isaaclab_flat_robot_cfg().replace(prim_path="{ENV_REGEX_NS}/Robot")
        cfg.action_scale = 0.25
        hidden_dims = ISL_FLAT_HIDDEN_DIMS
        print("[go2_play] Modo --isaaclab_flat: action_scale=0.25, rede=[128,128,128], "
              "coxa traseira=1.0, DCMotor, go2.usd do IsaacLab.")

    env = Go2PlayEnv(cfg=cfg, render_mode=None)
    print("[go2_play] Joints:", env.robot.data.joint_names)

    policy = load_policy(
        args_cli.checkpoint,
        num_obs=cfg.observation_space,
        num_actions=cfg.action_space,
        device=args_cli.device,
        hidden_dims=hidden_dims,
    )
    keyboard = KeyboardCommand()

    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    print("[go2_play] PRONTO — janela em foco: WASD anda, Q/E gira, espaço para.")

    while simulation_app.is_running():
        # injeta o comando de teclado em TODOS os ambientes
        vx, vy, yaw = keyboard.get()
        env._commands[:, 0] = vx
        env._commands[:, 1] = vy
        env._commands[:, 2] = yaw

        with torch.inference_mode():
            actions = policy(obs)

        obs_dict, _, _, _, _ = env.step(actions)
        obs = obs_dict["policy"]

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
