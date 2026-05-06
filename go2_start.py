# =============================================================================
# go2_start.py  —  go2_start + WAREHOUSE ENVIRONMENT
# Roda: C:\opt\GRVA\NVIDIA\isaacsim5\python.bat C:\W9\SRC2\scripts5\go2_start.py
# =============================================================================

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import carb, carb.input
import numpy as np
import omni.usd, omni.appwindow
from isaacsim.core.api import World
from isaacsim.core.api.robots import Robot
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction
from pxr import UsdGeom, Gf, UsdPhysics, UsdShade, Sdf

# Importações adicionais para carregar o ambiente Warehouse
from isaacsim.core.utils.prims import define_prim
from isaacsim.storage.native import get_assets_root_path

# =============================================================================
# CONFIGURAÇÕES GLOBAIS
# =============================================================================
GO2_USD  = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Robots/Unitree/Go2/go2.usd"
GO2_PRIM = "/World/Go2"

# Parâmetros de Simulação
PHYSICS_DT = 1/400.0  # Frequência de física (maior = mais estável)
RENDERING_DT = 1/60.0 # Frequência de renderização

# Parâmetros de Atrito
GROUND_FRICTION = 2.5 # Atrito estático e dinâmico do chão
FEET_FRICTION = 2.5   # Atrito estático e dinâmico das patas do robô
RESTITUTION = 0.0     # Coeficiente de restituição (elasticidade)

# Parâmetros do Controlador de Junta (KP/KD)
JOINT_KP = 200000.0   # Ganho Proporcional (Stiffness)
JOINT_KD = 4000.0     # Ganho Derivativo (Damping)

# Posição de Repouso (Stand Pose)
POSE_STAND = np.array([
     0.1, -0.1,  0.1, -0.1,  # Posições dos quadris (hip)
     0.8,  0.8,  0.8,  0.8,  # Posições das coxas (thigh)
    -1.5, -1.5, -1.5, -1.5,  # Posições das panturrilhas (calf)
], dtype=np.float64)

# Parâmetros da Marcha (Gait)
STRIDE_AMP = 0.10     # Amplitude do passo (forward/backward)
LIFT_AMP   = 0.08     # Amplitude de levantamento da pata
GAIT_FREQ  = 2.0      # Frequência da marcha
LATERAL_HIP_AMP = 0.03 # Amplitude do movimento lateral do quadril

# Parâmetros de Inicialização
INITIAL_STAND_STEPS = 400 # Número de passos de física para o robô ficar em pé inicialmente

# Parâmetros de Navegação por Alvo
TARGET_DISTANCE = 10.0 # Distância para o robô caminhar à frente (em metros)
TARGET_REACH_THRESHOLD = 0.2 # Distância para considerar o alvo alcançado (em metros)
FORWARD_SPEED_FACTOR = 0.5 # Fator de velocidade para o movimento em direção ao alvo

# Posição inicial do robô no ambiente Warehouse
GO2_INITIAL_POSITION = np.array([0, 0, 0.7]) # Ajustado para o ambiente Warehouse

# =============================================================================
# INICIALIZAÇÃO DO AMBIENTE ISAAC SIM
# =============================================================================
# Cria o aplicativo de simulação com a opção de modo headless
world = World(stage_units_in_meters=1.0, physics_dt=PHYSICS_DT, rendering_dt=RENDERING_DT)

# Carrega o ambiente Warehouse
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Não foi possível encontrar a pasta de assets do Isaac Sim.")

# Define o prim para o ambiente Warehouse e adiciona a referência ao asset USD
prim = define_prim("/World/Warehouse", "Xform")
warehouse_asset_path = assets_root_path + "/Isaac/Environments/Simple_Warehouse/warehouse.usd"
prim.GetReferences().AddReference(warehouse_asset_path)

# Adiciona o modelo USD do robô Go2 ao cenário
add_reference_to_stage(usd_path=GO2_USD, prim_path=GO2_PRIM)

# Obtém o stage atual do USD, que representa a cena da simulação
stage = omni.usd.get_context().get_stage()

# =============================================================================
# FUNÇÕES DE CONFIGURAÇÃO DE ATRITO
# =============================================================================
def set_ground_friction(stage, friction, restitution):
    """Aplica material de física com atrito no ground plane."""
    mat_path = "/World/GroundPhysicsMaterial"
    mat = UsdShade.Material.Define(stage, mat_path)
    phys_mat = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    phys_mat.GetStaticFrictionAttr().Set(friction)
    phys_mat.GetDynamicFrictionAttr().Set(friction)
    phys_mat.GetRestitutionAttr().Set(restitution)

    # Tenta aplicar o material a candidatos comuns de ground plane
    ground_candidates = ["/World/defaultGroundPlane", "/World/GroundPlane",
                         "/World/defaultGroundPlane/CollisionMesh",
                         "/World/defaultGroundPlane/CollisionPlane",
                         "/World/Warehouse/SM_Warehouse_Floor_01"]
    applied = 0
    for path in ground_candidates:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            binding = UsdShade.MaterialBindingAPI.Apply(prim)
            binding.Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")
            applied += 1
            carb.log_warn(f"[GO2] Atrito {friction} aplicado em: {path}")

    # Busca genérica se não achou em candidatos específicos
    if applied == 0:
        for prim in stage.Traverse():
            p = str(prim.GetPath())
            if "ground" in p.lower() or "plane" in p.lower() or "floor" in p.lower():
                if prim.HasAPI(UsdPhysics.CollisionAPI) or prim.IsA(UsdGeom.Mesh):
                    binding = UsdShade.MaterialBindingAPI.Apply(prim)
                    binding.Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")
                    carb.log_warn(f"[GO2] Atrito aplicado em: {p}")
                    applied += 1
    carb.log_warn(f"[GO2] Total: atrito aplicado em {applied} prims do chão")

def set_feet_friction(stage, robot_path, friction, restitution):
    """Aplica atrito nas patas do robô (calflower = pés do Go2)."""
    mat_path = "/World/FeetPhysicsMaterial"
    mat = UsdShade.Material.Define(stage, mat_path)
    phys_mat = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    phys_mat.GetStaticFrictionAttr().Set(friction)
    phys_mat.GetDynamicFrictionAttr().Set(friction)
    phys_mat.GetRestitutionAttr().Set(restitution)

    # Palavras-chave para identificar os prims das patas
    foot_kws = ["calflower", "foot", "hoof", "toe", "calf_lower"]
    applied = 0
    for prim in stage.Traverse():
        p = str(prim.GetPath())
        if not p.startswith(robot_path): continue # Ignora prims fora do robô
        pname = p.lower()
        if any(k in pname for k in foot_kws): # Verifica se o nome contém palavra-chave de pata
            if prim.HasAPI(UsdPhysics.CollisionAPI) or prim.IsA(UsdGeom.Mesh):
                binding = UsdShade.MaterialBindingAPI.Apply(prim)
                binding.Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")
                applied += 1
    carb.log_warn(f"[GO2] Atrito {friction} aplicado em {applied} prims de pata")

# Aplica as configurações de atrito
set_ground_friction(stage, friction=GROUND_FRICTION, restitution=RESTITUTION)
set_feet_friction(stage, GO2_PRIM, friction=FEET_FRICTION, restitution=RESTITUTION)

# =============================================================================
# CONFIGURAÇÃO DO ROBÔ
# =============================================================================
# Detecta o ArticulationRoot do robô para controle
art_prim = GO2_PRIM
for prim in stage.Traverse():
    p = str(prim.GetPath())
    if p.startswith(GO2_PRIM) and prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        art_prim = p
        break

# Posiciona o robô na cena
xform = UsdGeom.Xformable(stage.GetPrimAtPath(GO2_PRIM))
xform.ClearXformOpOrder()
# Define a posição inicial do robô (ajustado para o ambiente Warehouse)
xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(GO2_INITIAL_POSITION[0], GO2_INITIAL_POSITION[1], GO2_INITIAL_POSITION[2]))

# Configura os acionadores das juntas (Joint Drives)
count = 0
for prim in stage.Traverse():
    p = str(prim.GetPath())
    if not p.startswith(GO2_PRIM): continue
    if not prim.IsA(UsdPhysics.RevoluteJoint): continue
    
    # Remove drives existentes para evitar duplicação
    if prim.HasAPI(UsdPhysics.DriveAPI):
        prim.RemoveAPI(UsdPhysics.DriveAPI, "angular")
    
    # Aplica um novo drive de força angular
    drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
    drive.GetTypeAttr().Set("force")
    drive.GetStiffnessAttr().Set(JOINT_KP) # Define o ganho KP
    drive.GetDampingAttr().Set(JOINT_KD)   # Define o ganho KD
    drive.GetMaxForceAttr().Set(1e9)       # Força máxima permitida
    drive.GetTargetPositionAttr().Set(0.0) # Posição alvo inicial (neutra)
    count += 1
carb.log_warn(f"[GO2] {count} juntas configuradas com kp={JOINT_KP} kd={JOINT_KD}")

# Adiciona o robô ao mundo da simulação
go2 = world.scene.add(Robot(prim_path=art_prim, name="go2"))
world.reset()

# Obtém o número de graus de liberdade (DoF) e nomes das juntas
n_dof = go2.num_dof
names = go2.dof_names

# Define a posição inicial das juntas e zera as velocidades
go2.set_joint_positions(POSE_STAND)
go2.set_joint_velocities(np.zeros(n_dof))

# Configura os ganhos KP e KD para o controlador de articulação
go2.get_articulation_controller().set_gains(
    kps=np.full(n_dof, JOINT_KP),
    kds=np.full(n_dof, JOINT_KD)
)

# =============================================================================
# CONTROLE VIA TECLADO
# =============================================================================
_input    = carb.input.acquire_input_interface()
_keyboard = omni.appwindow.get_default_app_window().get_keyboard()
keys_state = {} # Dicionário para armazenar o estado das teclas pressionadas

def on_key(event, *args, **kwargs):
    """Callback para eventos de teclado."""
    try:
        k = event.input
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:   keys_state[k] = True
        if event.type == carb.input.KeyboardEventType.KEY_RELEASE: keys_state[k] = False
    except: pass
    return True

# Assina eventos de teclado
_sub = _input.subscribe_to_keyboard_events(_keyboard, on_key)

# Mapeamento de teclas para comandos de movimento (fwd, lat)
FMAP = {
    carb.input.KeyboardInput.UP:    ( 1, 0),
    carb.input.KeyboardInput.DOWN:  (-1, 0),
    carb.input.KeyboardInput.LEFT:  ( 0, 1),
    carb.input.KeyboardInput.RIGHT: ( 0,-1),
    carb.input.KeyboardInput.W:     ( 1, 0),
    carb.input.KeyboardInput.S:     (-1, 0),
    carb.input.KeyboardInput.A:     ( 0, 1),
    carb.input.KeyboardInput.D:     ( 0,-1),
}

def get_command():
    """Retorna os comandos de movimento (fwd, lat) baseados no teclado."""
    fwd, lat = 0.0, 0.0
    # Verifica o estado das teclas pressionadas
    for k, v in list(keys_state.items()):
        if v and k in FMAP:
            fwd += FMAP[k][0]; lat += FMAP[k][1]
    
    # Fallback para leitura direta do teclado (útil se keys_state não capturar tudo)
    if fwd == 0 and lat == 0:
        try:
            if _input.get_keyboard_value(_keyboard, carb.input.KeyboardInput.W)     > 0.5: fwd += 1
            if _input.get_keyboard_value(_keyboard, carb.input.KeyboardInput.S)     > 0.5: fwd -= 1
            if _input.get_keyboard_value(_keyboard, carb.input.KeyboardInput.A)     > 0.5: lat += 1
            if _input.get_keyboard_value(_keyboard, carb.input.KeyboardInput.D)     > 0.5: lat -= 1
            if _input.get_keyboard_value(_keyboard, carb.input.KeyboardInput.UP)    > 0.5: fwd += 1
            if _input.get_keyboard_value(_keyboard, carb.input.KeyboardInput.DOWN)  > 0.5: fwd -= 1
            if _input.get_keyboard_value(_keyboard, carb.input.KeyboardInput.LEFT)  > 0.5: lat += 1
            if _input.get_keyboard_value(_keyboard, carb.input.KeyboardInput.RIGHT) > 0.5: lat -= 1
        except: pass
    
    # Limita os comandos entre -1 e 1
    return float(np.clip(fwd,-1,1)), float(np.clip(lat,-1,1))

# =============================================================================
# CONTROLE DE MARCHA (GAIT) E NAVEGAÇÃO POR ALVO
# =============================================================================
step_count = 0
gait_t     = 0.0

# Variáveis para o controle de navegação por alvo
target_x = None
target_y = None

def on_physics_step(dt):
    """Callback executado a cada passo de física da simulação."""
    global step_count, gait_t, target_x, target_y
    step_count += 1

    # Fase de inicialização: robô fica em pé
    if step_count < INITIAL_STAND_STEPS:
        go2.get_articulation_controller().apply_action(
            ArticulationAction(joint_positions=POSE_STAND, joint_velocities=np.zeros(n_dof))
        )
        if step_count == INITIAL_STAND_STEPS - 1:
            carb.log_warn("[GO2] PRONTO! Use WASD/setas para andar ou 'T' para definir um alvo.")
        return

    # Obtém a posição atual do robô
    current_position = go2.get_world_pose()[0] # [0] para posição, [1] para orientação
    fwd, lat = get_command()

    # Lógica para definir um alvo (ex: 10 metros à frente)
    if carb.input.KeyboardInput.T in keys_state and keys_state[carb.input.KeyboardInput.T]:
        if target_x is None: # Define o alvo apenas uma vez ao pressionar 'T'
            # Define o alvo 10 metros à frente na direção X do robô
            target_x = current_position[0] + TARGET_DISTANCE
            target_y = current_position[1] # Mantém a coordenada Y atual
            carb.log_warn(f"[GO2] Alvo definido em: ({target_x:.2f}, {target_y:.2f})")
        keys_state[carb.input.KeyboardInput.T] = False # Reseta o estado da tecla

    # Se um alvo estiver definido, o robô tenta alcançá-lo
    if target_x is not None:
        distance_to_target = np.sqrt((target_x - current_position[0])**2 + (target_y - current_position[1])**2)
        
        if distance_to_target < TARGET_REACH_THRESHOLD:
            carb.log_warn("[GO2] Alvo alcançado!")
            target_x = None # Reseta o alvo
            fwd = 0.0 # Para o robô
            lat = 0.0
        else:
            # Calcula a direção para o alvo (simplificado para movimento em X)
            # Para um controle mais avançado, seria necessário considerar a orientação do robô
            if target_x > current_position[0]:
                fwd = FORWARD_SPEED_FACTOR
            else:
                fwd = -FORWARD_SPEED_FACTOR
            lat = 0.0 # Mantém o movimento lateral zero para navegação simples

    target_joint_positions = POSE_STAND.copy()

    # Aplica a marcha se houver comando de movimento (teclado ou alvo)
    if abs(fwd) > 0.01 or abs(lat) > 0.01:
        gait_t += dt * GAIT_FREQ * 2 * np.pi
        pA = np.sin(gait_t)        # Fase para patas FL (Front-Left) e RR (Rear-Right)
        pB = np.sin(gait_t + np.pi) # Fase para patas FR (Front-Right) e RL (Rear-Left)

        # Movimento lateral dos quadris
        target_joint_positions[0] += lat * LATERAL_HIP_AMP    # FL hip
        target_joint_positions[1] -= lat * LATERAL_HIP_AMP    # FR hip
        target_joint_positions[2] += lat * LATERAL_HIP_AMP    # RL hip
        target_joint_positions[3] -= lat * LATERAL_HIP_AMP    # RR hip

        # Movimento das coxas (thighs) para frente/trás
        for idx, ph in [(4,pA),(5,pB),(6,pB),(7,pA)]:   # Índices das coxas
            target_joint_positions[idx] += ph * fwd * STRIDE_AMP
        
        # Levantamento das panturrilhas (calfs) para simular o passo
        for idx, ph in [(8,pA),(9,pB),(10,pB),(11,pA)]:  # Índices das panturrilhas
            target_joint_positions[idx] += max(ph, 0) * LIFT_AMP # Levanta apenas na fase positiva

    # Aplica a ação de articulação com as posições alvo das juntas
    go2.get_articulation_controller().apply_action(
        ArticulationAction(joint_positions=target_joint_positions, joint_velocities=np.zeros(n_dof))
    )

# Adiciona o callback de física ao mundo da simulação
world.add_physics_callback("go2_step", on_physics_step)

carb.log_warn("[GO2] Iniciando go2_start — com ambiente Warehouse e navegação por alvo...")

# Loop principal da simulação
while simulation_app.is_running():
    world.step(render=True) # Avança um passo na simulação e renderiza

simulation_app.close() # Fecha o aplicativo de simulação ao sair do loop
