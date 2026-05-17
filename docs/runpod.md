# RunPod cluster wiring

How the local `pi-stack` repo plugs into a RunPod GPU cluster. This is the
narrative companion to §8 of [`CHECKLIST.md`](../CHECKLIST.md) — same steps,
more context.

> **TL;DR.** Two pods (dev + training), one persistent network volume, git
> for code sync. The same `pi-stack` code runs both places; only the
> `backbone=` argument changes (`TinyBackbone` locally, `Gemma3 4B` on a
> pod).

## The split

| | **Local laptop** | **RunPod dev pod** | **RunPod training pod** |
|---|---|---|---|
| GPU | None / Apple MPS | 1× A100 80 GB or H100 PCIe | 8× H100 SXM (NVLink) |
| Cost | $0 | ~$1.5-3 /hr | ~$25-35 /hr |
| Purpose | Edit + `pytest` inner loop | Real-backbone fine-tunes, RL, eval | Full pre-training only |
| Lifecycle | Always on | Stop when idle | Spin up for training campaigns |
| What runs here | `TinyBackbone` policy code, MEM/RLT/RECAP mechanics, RTC algorithm, FAST tokenizer | Real PaliGemma / Gemma 3 inference, KI fine-tune, RECAP RL, RLT online refinement, real Libero rollouts | Multi-GPU pre-training of π₀, π₀.₅, π*₀.₆, π₀.₇ |

The shared **Network Volume** (~1-2 TB) is mounted at `/workspace` on every
pod and persists across pod restarts. Datasets, checkpoints, and the HF
cache live there.

## Why two pods, not one

A 1× A100 dev pod at $2/hr running 8 hours a day, 20 days a month is
~$320/month. An always-on 8× H100 SXM pod is ~$2.4k/day — overkill for
fine-tuning, eval, and the inner-loop work that dominates the project. The
training pod only earns its keep during actual pre-training campaigns, so
you spin it up on demand and shut it down between runs.

## Walkthrough

### 1. Account setup (one-time)

1. Sign up at https://www.runpod.io. Add a payment method.
2. **Settings → API Keys** — generate a key, store it in 1Password.
3. **Settings → SSH Public Keys** — paste your laptop's `~/.ssh/id_ed25519.pub`.
4. (optional) `pip install runpod` locally for scripted pod control.

### 2. Provision the network volume (one-time)

In the RunPod console, **Storage → Network Volumes → Create Volume**:
- Size: 1 TB to start; resize up if you ingest more OXE data
- Region: pick the one where your pods will run (same region as the pod or
  the volume won't attach)

You'll attach this same volume to both pod templates below.

### 3. Build a container image (one-time, ~20 min)

We bake the heavy `uv sync` into the image so cold-starting a new pod is
seconds rather than minutes. Skeleton `Dockerfile`:

```dockerfile
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4-devel

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs build-essential libgl1 ffmpeg ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

# uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
 && ln -s /root/.local/bin/uv /usr/local/bin/uv

# Pre-resolve our deps (so the network volume's pi-stack just `uv sync`s instantly)
WORKDIR /opt/pi-stack-prefetch
COPY pyproject.toml uv.lock ./
RUN uv sync --extra ml --extra dev --extra sim --extra jax --extra oxe

# Default workdir is the network volume mount.
WORKDIR /workspace
```

Build and push:

```bash
docker build -t ghcr.io/<you>/pi-stack-runpod:latest .
docker push ghcr.io/<you>/pi-stack-runpod:latest
```

### 4. Create pod templates

In the RunPod console, **Templates → New Template**:

**Dev template:**
- Image: `ghcr.io/<you>/pi-stack-runpod:latest`
- GPU: 1× A100 80GB PCIe (or H100 PCIe for ~2× speed at ~2× cost)
- Volume mount: your network volume at `/workspace`
- Expose ports: 22 (SSH), 8888 (JupyterLab), 8765 (the RTC inference server)
- Environment variables: `HF_HOME=/workspace/hf-cache`, `WANDB_DIR=/workspace/wandb`

**Training template:**
- Same image, same volume, same env vars
- GPU: **8× H100 SXM (NVLink)** — this is the pre-training spec from the PI papers
- Spot pricing if your trainer checkpoints and resumes cleanly

### 5. First pod boot

Start the dev pod. SSH in:

```bash
ssh root@<pod-public-ip> -p <pod-ssh-port>
```

On the pod:

```bash
cd /workspace
git clone <your-fork-url> pi-stack
cd pi-stack
uv sync --extra ml --extra dev --extra sim --extra jax --extra oxe
bash scripts/setup_sim.sh
uv run pytest -q          # expect 65/65
export HF_TOKEN=<paste your token>
huggingface-cli whoami    # expect your username
```

The first `uv sync` reuses the pre-resolved deps from the image, so it
takes seconds. From here, the code runs identically to your laptop —
except now `load_backbone(GEMMA3_4B)` will actually work.

### 6. Wire up the real backbone

`pi_stack.models.backbones.load_backbone()` currently raises
`NotImplementedError` for the real models. On the pod, replace its body
with a real `transformers` load that returns a thin adapter exposing
`(logits, features)` matching `TinyBackbone`'s contract. Sketch:

```python
def load_backbone(spec, *, device=None):
    if spec.name == "tiny":
        return TinyBackbone(hidden_size=spec.hidden_size, vocab_size=spec.vocab_size)
    if spec.name == "gemma3":
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            spec.hf_repo, torch_dtype=torch.bfloat16, device_map=device or "cuda"
        )
        return _Gemma3Adapter(model)   # exposes .net.patch_proj, returns (logits, features)
    ...
```

Smoke-test:

```bash
uv run python -c "
from pi_stack.models.backbones import GEMMA3_4B, load_backbone
from pi_stack.models.pi07 import Pi07Config, Pi07Policy
import torch

backbone = load_backbone(GEMMA3_4B, device='cuda')
policy = Pi07Policy(Pi07Config(backbone=GEMMA3_4B), backbone=backbone)
images = torch.randn(1, 3, 224, 224).cuda().to(torch.bfloat16)
state = torch.randn(1, 14).cuda().to(torch.bfloat16)
ids = torch.randint(0, GEMMA3_4B.vocab_size, (1, 16)).cuda()
print(policy.predict_chunk(images, state, ids).shape)
"
```

### 7. Hydrate datasets onto the volume

Libero and Kinetix install via `scripts/setup_sim.sh` (already idempotent).
OXE requires data download — to keep it on the persistent volume:

```bash
export TFDS_DATA_DIR=/workspace/data/oxe
uv run python -c "
import tensorflow_datasets as tfds
tfds.load('fractal20220817_data', data_dir=tfds.core.constants.DATA_DIR)
"
```

The first call downloads to `/workspace/data/oxe/`; subsequent pod
restarts reuse it.

### 8. Multi-GPU training

For the 8× H100 training pod, use `accelerate` as the launcher:

```bash
accelerate config       # one-time per pod template; pick FSDP or DDP
accelerate launch scripts/train.py --config configs/pi07.yaml
```

`scripts/train.py` is still a stub today — wiring `accelerate` through
its trainer-dispatch logic is one of the §8h checklist items.

### 9. Inference / robot deployment

`scripts/infer_server.py` serves chunks over WebSocket. On the inference
pod (or the dev pod if you don't want a third template):

```bash
uv run python scripts/infer_server.py --config configs/rtc.yaml
```

Expose port 8765 in the RunPod console. The robot client connects to
`ws://<pod-ip>:8765` and streams obs → actions. RTC absorbs the 50-200 ms
network round-trip; that's the whole point of §3f.

### 10. Stopping the pod

Two paths:

1. **Stop** — pod is frozen, volume persists, hourly billing pauses. Restart
   to resume in seconds.
2. **Terminate** — pod is destroyed; the volume still persists (it's a
   separate resource). Cheapest if you won't be back for days.

Default to **stop** on the dev pod. **Terminate** the training pod between
runs — there's no point paying for an idle H100 cluster.

## Common gotchas

- **Volume region mismatch.** A volume in `us-east-1` won't attach to a pod
  in `eu-central-1`. Pick one region and stick to it.
- **HF cache not persisting.** Make sure `HF_HOME=/workspace/hf-cache` is
  set as a template env var — otherwise the cache lands in container-local
  storage that vanishes on pod termination.
- **Spot pod preemption.** If your trainer doesn't checkpoint frequently,
  spot pricing is a trap. Wire `--save_every_n_steps` first, then enable
  spot.
- **Idle billing.** Stopped pods don't cost compute, but the volume costs
  ~$0.10/GB/month regardless.

## See also

- §8 of [`CHECKLIST.md`](../CHECKLIST.md) — checkbox version of this guide.
- [`docs/sim-setup.md`](./sim-setup.md) — sim-stack install steps (same on
  both environments).
- [`CLAUDE.md`](../CLAUDE.md) — what work belongs local vs on the pod.
