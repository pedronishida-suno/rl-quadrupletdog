# Demo — Go2 no mapa HUSF 2026

Passo a passo para rodar o **Unitree Go2** com política treinada andando dentro do
mapa **HUSF 2026** (`HUSF 2026.usdc`), usando `go2_play_map.py`.

> Rode tudo a partir da **raiz do IsaacLab** (a pasta com `isaaclab.sh` / `isaaclab.bat`),
> com `go2_play_map.py`, `go2_rl_env.py` e `go2_rl_train.py` no mesmo diretório.

---

## Pré-requisitos

- Isaac Sim 5.1 + IsaacLab + RSL-RL instalados.
- O arquivo `HUSF 2026.usdc` acessível no disco (anote o caminho completo).
- Uma GPU com VRAM suficiente — o mapa tem ~460 MB; numa 6 GB roda apertado, use `--num_envs 1`.

---

## Passo 1 — Obter o checkpoint do Go2

Não há checkpoint no repositório (`.pt` é ignorado pelo `.gitignore`). Use o **checkpoint
oficial pré-treinado** do IsaacLab. Rode uma vez — ele baixa e mostra no log a linha
`Loading model checkpoint from: <caminho>.pt`. **Copie esse caminho.**

**Linux**
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Velocity-Flat-Unitree-Go2-v0 --num_envs 1 --use_pretrained_checkpoint
```

**Windows**
```bat
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\play.py ^
    --task Isaac-Velocity-Flat-Unitree-Go2-v0 --num_envs 1 --use_pretrained_checkpoint
```

> Já tem um checkpoint treinado por você (`go2_rl_train.py`)? Pule este passo, use o seu `.pt`
> e **não** passe `--isaaclab_flat` no passo 2.

---

## Passo 2 — Rodar o Go2 no mapa HUSF

**Linux** (aspas por causa do espaço no nome do arquivo)
```bash
./isaaclab.sh -p go2_play_map.py --isaaclab_flat \
    --checkpoint <caminho-do-.pt-do-passo-1> \
    --map "/caminho/para/HUSF 2026.usdc" --num_envs 1
```

**Windows**
```bat
isaaclab.bat -p go2_play_map.py --isaaclab_flat ^
    --checkpoint <caminho-do-.pt-do-passo-1> ^
    --map "C:\caminho\para\HUSF 2026.usdc" --num_envs 1
```

O `--isaaclab_flat` casa o robô com o checkpoint oficial: rede `[128,128,128]`,
`action_scale=0.25`, coxa traseira `1.0`, atuador `DCMotor` e o `go2.usd` do IsaacLab
(mesma ordem de juntas do treino).

### Controles (janela do Isaac em foco)

| Tecla | Ação |
|---|---|
| `W` / `S` | Frente / trás |
| `A` / `D` | Esquerda / direita |
| `Q` / `E` | Girar esq. / dir. |
| `espaço`  | Parar |

---

## Ajustes que podem ser necessários ao vivo

| Situação | O que fazer |
|---|---|
| Robô flutua ou afunda em relação ao chão do mapa | `--map_z <valor>` (ex.: `--map_z -0.5`) até o piso do HUSF coincidir com o robô |
| Mapa em escala errada | `--map_scale <fator>` — deixe `1.0` para o HUSF (USD nativo em metros) |
| OOM / travamento | É VRAM. Confirme `--num_envs 1`; numa 6 GB o mapa é pesado |
| Robô nasce dentro de uma parede e treme | Mova o spawn ou abra o mapa numa área livre |

> **Vantagem do `go2_play_map.py`:** ele sempre cria um **piso plano com colisão** por baixo
> e coloca o HUSF por cima. Então, mesmo que o `.usdc` seja só visual (sem malha de colisão),
> o Go2 tem onde pisar — diferente do script standalone do v15, em que o robô cairia.

---

## Roteiro seguro para a apresentação

1. **Verificar o mapa:** rode o script standalone do v15 com ANYmal apontando para o `.usdc` real.
   Se o ANYmal não cair, o HUSF tem colisão (e o Go2 também fica de pé).
2. **Mostrar o Go2:** rode o Passo 2 — é o resultado-alvo.
3. **Rede de segurança:** se o checkpoint adaptado tremer/cair, o Passo 1 sozinho já mostra um
   Go2 treinado andando em chão plano, sem depender do nosso código.

---

## Troubleshooting

| Erro | Causa provável |
|---|---|
| `No module named 'isaaclab'` | Use `isaaclab.sh -p` / `isaaclab.bat -p`, não o Python do sistema |
| `state_dict` não carrega / tamanhos não batem | Faltou `--isaaclab_flat` (rede [128,128,128] vs [512,256,128]) |
| Robô treme e cai logo de cara | Suspeito nº 1: ordem das juntas. Compare a lista impressa em `[go2_play] Joints:` com o `env.yaml` do checkpoint |
| Robô atravessa o chão | Use `--map_z` para alinhar; o piso plano embutido deveria evitar isso |

> ⚠️ **Não testado na máquina onde este repo foi editado** (Isaac não instalado lá).
> Faça um teste antes da reunião.
