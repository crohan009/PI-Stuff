# pi-stack

Single-handed implementation of the **Physical Intelligence** research arc — from
**π₀** (Oct 2024) to **π₀.₇** (Apr 2026). One Python package, one submodule per
paper, traceable from PDF to code.

> Full arc summary: [`About-PI.md`](./About-PI.md). Paper PDFs: [`papers/`](./papers/).
> Live status: [`CHECKLIST.md`](./CHECKLIST.md). Sim/eval install: [`docs/sim-setup.md`](./docs/sim-setup.md).

## What's in the box

| Paper | Module |
|---|---|
| π₀ — first generalist policy (PaliGemma + flow matching) | `pi_stack.models.pi0` |
| FAST — DCT+BPE action tokenization | `pi_stack.tokenization.fast` |
| Hi Robot — System 1 / System 2 hierarchy | `pi_stack.models.hi_robot` |
| π₀.₅ — open-world generalization, co-training | `pi_stack.models.pi05` |
| KI — knowledge insulation training recipe | `pi_stack.training.ki` |
| RTC — real-time chunking (async inpainting) | `pi_stack.inference.rtc` |
| π*₀.₆ — RECAP RL from experience | `pi_stack.models.pi06` + `pi_stack.training.recap` |
| Human-to-Robot — egocentric video transfer | `pi_stack.data.human_to_robot` |
| MEM — short-term video + long-term language memory | `pi_stack.memory.mem` |
| RLT — RL-token local refinement | `pi_stack.rlt.rl_token` |
| π₀.₇ — steerable generalist (Gemma 3 + multi-modal context) | `pi_stack.models.pi07` |

Each module's docstring cites the paper section it implements.

## Quick start

```bash
# 1. Install uv (if you haven't). Lands in ~/.local/bin.
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Make sure ~/.local/bin is on PATH for this shell:
export PATH="$HOME/.local/bin:$PATH"

# 3. Sync the bare environment (Python 3.11; uv picks it up from pyenv
#    or downloads one automatically).
uv sync

# 4. Pull in the ML stack (transformers, accelerate, scipy, sentencepiece,
#    and torch+MPS/CPU — on macOS arm64 you get MPS automatically).
uv sync --extra ml

# 5. Add the dev tools and run smoke tests.
uv sync --extra ml --extra dev
uv run pytest -q

# 6. (Linux + NVIDIA only) replace the default torch with a CUDA build:
uv pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision
```

Everything heavier than `numpy` is gated behind extras (`ml`, `gpu`, `jax`,
`sim`, `serve`, `track`, `dev`, `all`) so the bare install stays light.

> `transformers` is currently pinned to `<5` — the PI-published FAST
> tokenizer's custom code on HF Hub is authored against the v4 tokenizer
> API. Revisit when PI ships a v5-compatible build.

## Activating and deactivating the environment

uv creates a project-local venv at `.venv/` on first `uv sync`. There are
two equivalent ways to use it:

### A) `uv run <cmd>` (no activation needed — recommended for one-offs)

```bash
uv run python scripts/train.py --config configs/pi0.yaml
uv run pytest -q
uv run python -c "import torch; print(torch.backends.mps.is_available())"
```

`uv run` always uses `.venv/`, regardless of what's "active." This is the
right pattern for CI, Make targets, and most scripts.

### B) Activate the venv as your shell's default (recommended for interactive sessions)

```bash
# Activate (zsh / bash):
source .venv/bin/activate
# You'll see (pi-stack) prepended to your prompt.

# Now `python`, `pytest`, `pip`, etc. all resolve to .venv/bin/*:
python scripts/eval.py --model configs/pi0.yaml --eval configs/eval/libero.yaml
pytest -q
which python   # -> /Users/.../PI-Stuff/.venv/bin/python

# Deactivate when done:
deactivate
```

Fish shell users: `source .venv/bin/activate.fish`. PowerShell: `.\.venv\Scripts\Activate.ps1`.

### Re-syncing after pulling new changes

```bash
uv sync --extra ml --extra dev       # match whatever extras you use
```

uv is idempotent — running sync when the lockfile matches the venv is a
no-op (well under a second).

### Nuking the venv and starting over

```bash
rm -rf .venv uv.lock
uv sync --extra ml --extra dev
```

The `.venv/` directory is gitignored, so this is always safe.

## Repository layout

```
PI-Stuff/
├── About-PI.md           # research arc summary (start here)
├── papers/               # local PDFs of all 11 PI papers + cited prior work
├── src/pi_stack/         # the package — one submodule per paper
│   ├── models/           # pi0, pi05, pi06, pi07, hi_robot
│   ├── tokenization/     # fast (DCT + BPE)
│   ├── training/         # flow_matching, ki, recap
│   ├── inference/        # rtc, server (WebSocket)
│   ├── memory/           # mem (short + long term)
│   ├── rlt/              # rl_token (online RL refinement)
│   ├── data/             # chunking, synthetic, human_to_robot
│   ├── envs/             # libero, kinetix, mujoco wrappers
│   └── utils/
├── configs/              # YAML configs per model and recipe
├── scripts/              # train.py, eval.py, infer_server.py
├── tests/                # smoke tests
├── docs/                 # one design doc per paper, written as we implement
├── notebooks/            # exploration / scratch
├── checkpoints/          # local weights (gitignored)
└── data/                 # local datasets (gitignored)
```

## Hardware reality

- **Inference / fine-tuning / RL refinement:** one RTX 4090 is enough for π₀-class
  policies thanks to KI + FAST (≈ 7.5× faster training than pure diffusion).
- **Full pre-training of π₀.₆ / π₀.₇:** needs ≥ 8×H100. Treat as aspirational.
- **RTC:** designed to tolerate 300+ ms inference latency, so a remote
  workstation over LAN/WebSocket is a viable deployment.

## License

Apache-2.0 (placeholder — adjust before publishing).
