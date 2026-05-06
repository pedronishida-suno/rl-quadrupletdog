# Go2 — Isaac Sim & IsaacLab Scripts

Três scripts para o robô quadrúpede **Unitree Go2** no NVIDIA Isaac.

| Arquivo | Plataforma | O que faz |
|---|---|---|
| `go2_start.py` | Isaac Sim 5.1 | Simulação interativa com controle por teclado num ambiente Warehouse |
| `go2_rl_env.py` | IsaacLab | Define o ambiente RL (importado pelo script de treino) |
| `go2_rl_train.py` | IsaacLab | Treina locomoção com PPO via RSL-RL |

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
