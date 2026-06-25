# Go2 — Isaac Sim & IsaacLab Scripts

Três scripts para o robô quadrúpede **Unitree Go2** no NVIDIA Isaac.

| Arquivo | Plataforma | O que faz |
|---|---|---|
| `go2_start.py` | Isaac Sim 5.1 | Simulação interativa com controle por teclado num ambiente Warehouse |
| `go2_rl_env.py` | IsaacLab | Define o ambiente RL (importado pelos scripts de treino/playback) |
| `go2_rl_train.py` | IsaacLab | Treina locomoção com PPO via RSL-RL |
| `go2_play_map.py` | IsaacLab | Roda a **política treinada** num cenário 3D (built-in ou USD próprio), dirigida por teclado |

---

## Pré-requisitos

### `go2_start.py` — Isaac Sim

Requer **NVIDIA Isaac Sim 5.1** instalado (via Omniverse Launcher ou pacote `.deb`).

```bash
# Verificar instalação (Linux — caminhos comuns)
~/.local/share/ov/pkg/isaac-sim-5.1.*/python.sh --version
/opt/isaac-sim/python.sh --version
```

### `go2_rl_train.py` + `go2_rl_env.py` — IsaacLab

Requer **IsaacLab** + **RSL-RL**:

```bash
pip install rsl-rl
pip install isaaclab-rl  # opcional, mas recomendado
```

---

## Como rodar

### 1. `go2_start.py` — Simulação Interativa

Usa o Python **embutido do Isaac Sim**, não o Python do sistema.

```bash
# Linux (ajuste para sua instalação)
~/.local/share/ov/pkg/isaac-sim-5.1.*/python.sh go2_start.py
```

**Controles de teclado:**

| Tecla | Ação |
|---|---|
| `W` / `↑` | Andar para frente |
| `S` / `↓` | Andar para trás |
| `A` / `←` | Mover para esquerda |
| `D` / `→` | Mover para direita |
| `T` | Definir alvo 10 m à frente (navegação automática) |

O robô fica em pé durante os primeiros 400 passos antes de aceitar comandos.

---

### 2. `go2_rl_train.py` — Treino RL (PPO)

`go2_rl_env.py` é importado automaticamente — não execute diretamente. Os dois arquivos devem estar no mesmo diretório.

```bash
# Headless — máxima performance
isaaclab.sh -p go2_rl_train.py --headless --num_envs 4096

# Com GUI — menos ambientes para caber na VRAM
isaaclab.sh -p go2_rl_train.py --num_envs 512

# Retomar checkpoint
isaaclab.sh -p go2_rl_train.py --headless --num_envs 4096 \
    --resume logs/rsl_rl/go2_locomotion/YYYY-MM-DD_HH-MM-SS/model_500.pt
```

**Parâmetros:**

| Flag | Padrão | Descrição |
|---|---|---|
| `--num_envs` | `4096` | Ambientes paralelos (reduzir se faltar VRAM) |
| `--max_iterations` | `1500` | Iterações PPO |
| `--resume` | — | Checkpoint `.pt` para continuar treino |
| `--device` | `cuda:0` | Dispositivo PyTorch |
| `--headless` | off | Desativa GUI |

Checkpoints e logs salvos em `logs/rsl_rl/go2_locomotion/<timestamp>/`.

```bash
# Ver métricas em tempo real
tensorboard --logdir logs/rsl_rl/go2_locomotion
```

---

### 3. `go2_play_map.py` — Política treinada num cenário 3D

Carrega um checkpoint PPO e roda a política num ambiente 3D, com o Go2 dirigido por teclado.
A observação (48-dim), o `action_scale` e a pose padrão são reusados de `go2_rl_env.py`, então o
comportamento é idêntico ao do treino.

```bash
# cenário built-in Warehouse (mais seguro p/ demo)
isaaclab.sh -p go2_play_map.py \
    --checkpoint logs/rsl_rl/go2_locomotion/<run>/model_1500.pt

# outros cenários prontos do Isaac
isaaclab.sh -p go2_play_map.py --checkpoint <ckpt>.pt --map office     # office | hospital | warehouse_shelves | flat

# mapa PRÓPRIO (o da UFU / Santa Mônica depois de virar .usd)
isaaclab.sh -p go2_play_map.py --checkpoint <ckpt>.pt \
    --map /caminho/santa_monica.usd --map_scale 0.0254
```

#### Usando o checkpoint OFICIAL do IsaacLab (sem treinar)

O `--checkpoint` precisa de um `.pt`. Você pode treinar o seu (`go2_rl_train.py`) **ou**
usar o checkpoint oficial pré-treinado do NVIDIA IsaacLab. Eles **não** são compatíveis
por padrão (rede, `action_scale`, pose e atuador diferentes) — por isso existe a flag
`--isaaclab_flat`, que casa o robô/escala/rede exatamente com o checkpoint flat oficial.

```bash
# 1) Baixa o checkpoint oficial (roda 1x; veja no log a linha
#    "Loading model checkpoint from: <caminho .pt>")
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Velocity-Flat-Unitree-Go2-v0 --num_envs 1 --use_pretrained_checkpoint

# 2) Roda esse mesmo .pt no nosso script, no modo compatível, dentro do mapa
isaaclab.sh -p go2_play_map.py --isaaclab_flat \
    --checkpoint <caminho-do-.pt-do-passo-1> --map warehouse
```

O `--isaaclab_flat` aplica: rede `[128,128,128]`, `action_scale=0.25`, coxa traseira `1.0`,
atuador `DCMotor` e o `go2.usd` do IsaacLab (mesma ordem de juntas do treino).

> ⚠️ **Não testado nesta máquina** (Isaac não está instalado aqui). Faça um teste rápido na
> máquina onde o Isaac roda: se o robô tremer/cair de imediato, o suspeito nº 1 é a ordem
> das juntas — confira a lista impressa em `[go2_play] Joints:` contra o `env.yaml` do checkpoint.

**Controles de teclado** (janela do Isaac em foco):

| Tecla | Ação |
|---|---|
| `W` / `S` | Frente / trás (vx) |
| `A` / `D` | Esquerda / direita (vy) |
| `Q` / `E` | Girar esq. / dir. (yaw) |
| `espaço`  | Parar (zera o comando) |

| Flag | Padrão | Descrição |
|---|---|---|
| `--checkpoint` | — (obrigatório) | `.pt` treinado por `go2_rl_train.py` |
| `--map` | `warehouse` | Chave built-in ou caminho/URL de um `.usd` |
| `--map_scale` | `1.0` | Escala do mapa (use `0.0254` se export em polegadas) |
| `--map_z` | `0.0` | Deslocamento vertical do mapa p/ alinhar o piso em z=0 |
| `--isaaclab_flat` | off | Modo compatível com o checkpoint flat oficial do IsaacLab |

---

### Usar o mapa da UFU / Santa Mônica (`.skp` → `.usd`)

O Isaac **não lê `.skp`**. É preciso converter em dois passos:

**1. SketchUp → OBJ/FBX/glTF.** No SketchUp (desktop ou web), abra
`Maquete PLACA santa mônica Matheus.skp` e exporte para `.obj` (ou `.fbx`/`.gltf`).
SketchUp usa **polegadas** por padrão — anote isso para a escala (passo 3).

**2. OBJ → USD** com o conversor do IsaacLab:

```bash
isaaclab.sh -p scripts/tools/convert_mesh.py \
    santa_monica.obj santa_monica.usd \
    --collision-approximation convexDecomposition --make-instanceable
```

**3. Rodar** apontando `--map` para o `.usd`. Se o piso ficar gigante, o modelo veio em
polegadas → use `--map_scale 0.0254` (1 pol = 0,0254 m). Ajuste `--map_z` se o robô
nascer dentro/acima do chão.

> ⚠️ Mapas grandes/detalhados pesam na GPU. Numa RTX 3060 6 GB, rode com `--num_envs 1`
> e, se faltar VRAM, simplifique a malha no export.

---

## Detalhes do ambiente RL

| Campo | Valor |
|---|---|
| Observações | 48-dim: vel. linear, angular, gravidade, cmd, juntas, ações anteriores |
| Ações | 12-dim: deltas de posição de junta |
| Frequência de física | 200 Hz (política a 50 Hz) |
| Duração do episódio | 20 s |
| Terminação | altura < 0.08 m ou tempo esgotado |
| Fall recovery | 30% dos resets spawnam o robô de lado |

---

## Solução de problemas

**`No module named 'isaacsim'`** → Use `python.sh` do Isaac Sim, não o Python do sistema.

**`No module named 'isaaclab'`** → Use `isaaclab.sh -p` ou ative o ambiente do IsaacLab.

**`No module named 'rsl_rl'`** → `pip install rsl-rl`

**OOM na GPU** → Reduza `--num_envs 1024` ou `--num_envs 512`

**Sensores de contato não detectam patas** → O script imprime os nomes de corpo ao iniciar; verifique se contêm `_foot`. Se não, ajuste `contact_sensor_cfg` em `go2_rl_env.py`.
