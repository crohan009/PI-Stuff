# Master checklist — pi-stack

Single source of truth for what's done and what's next across implementation,
training, evaluation, and inference. Sorted roughly in dependency order
within each section so you can work top-down.

> **Update rule (read this).** Whenever a task in this file is finished, the
> corresponding line MUST be flipped from `- [ ]` to `- [x] (YYYY-MM-DD)` in
> the same change that completes the task. This rule is enforced via
> [`CLAUDE.md`](./CLAUDE.md#master-checklist) — Claude will not report a task
> as complete without updating this file. If a task is partially done, leave
> it unchecked and add a sub-bullet describing the remaining work.

Legend: `- [ ]` open · `- [x] (YYYY-MM-DD)` done · `- [~] (YYYY-MM-DD)` in progress / partial · `- [!]` blocked (note why)

---

## 0. Repo bootstrap

- [x] (2026-05-10) Scaffold directory tree (`src/pi_stack/`, `configs/`, `scripts/`, `tests/`, `docs/`)
- [x] (2026-05-10) `pyproject.toml` with extras: `ml`, `gpu`, `jax`, `sim`, `serve`, `track`, `dev`, `all`
- [x] (2026-05-10) `.gitignore` excluding `papers/*.pdf`, weights, datasets
- [x] (2026-05-10) `README.md` quick-start
- [x] (2026-05-10) `CLAUDE.md` with paper → module map
- [x] (2026-05-10) `docs/sim-setup.md` install guide for Libero / Kinetix / MuJoCo / OXE
- [x] (2026-05-10) `git init` + first commit on `main`
- [x] (2026-05-10) `CHECKLIST.md` (this file) with auto-update rule wired into CLAUDE.md
- [x] (2026-05-10) Common eval-env protocol (`pi_stack.envs.base.BaseEvalEnv`)

## 1. Local environment

- [x] (2026-05-17) Install `uv` (`curl -LsSf https://astral.sh/uv/install.sh | sh`) — uv 0.11.12, into `~/.local/bin`
- [x] (2026-05-17) `uv sync` succeeds (bare install) — Python 3.11.9 from pyenv, 17 core packages
- [x] (2026-05-17) `uv sync --extra ml` succeeds (transformers + accelerate + scipy + sentencepiece)
  - sub: added `sentencepiece` to the `ml` extra — required by the FAST processor's BPE backend
  - sub: pinned `transformers<5` — PI's published FAST tokenizer code on HF Hub is authored against v4 tokenizer API
- [x] (2026-05-17) `uv sync --extra dev` and `uv run pytest -q` passes smoke tests (6/6)
- [x] (2026-05-17) `HF_TOKEN` set in `.env` and `huggingface-cli whoami` works — logged in as `crohan009`, org `context-course`
- [x] (2026-05-17) GPU torch wheels installed for the local CUDA version (or MPS/CPU fallback verified on macOS) — torch 2.11.0 + MPS backend live on macOS-14.5-arm64; tensor on `mps:0` confirmed
- [x] (2026-05-17) `uv run python -c "from transformers import AutoProcessor; AutoProcessor.from_pretrained('physical-intelligence/fast', trust_remote_code=True)"` succeeds — `UniversalActionProcessor`, round-trip MSE 2.8e-4 on a (50, 14) sine chunk
- [ ] (optional) `wandb login` if using `track` extra

## 2. Simulation & evaluation stack

### 2a. Libero
- [x] (2026-05-17) Clone + `uv pip install -e ~/sim/libero` — automated via `scripts/setup_sim.sh`
  - sub: needed two local fixes captured in the script: empty `~/sim/libero/libero/__init__.py` (namespace package), and a pre-seeded `~/.libero/config.yaml` (skips the interactive first-run `input()` prompt)
- [x] (2026-05-17) Smoke: `from libero.libero import benchmark` imports
- [ ] `pi_stack.envs.libero.LiberoEnv.reset()` returns a real obs dict
- [ ] `step_chunk()` runs an H=50 chunk through robosuite
  - blocker: Libero's `requirements.txt` pins robosuite 1.4 + old numpy/transformers/gym which would clobber the ML stack; resolution deferred until we implement the wrapper
- [~] (2026-05-17) All 5 sub-suites instantiable (Spatial / Object / Goal / 10 / 90)
  - sub: 5 sub-suites + `libero_100` visible at the `benchmark_dict` layer; per-task instantiation needs robosuite (see above)

### 2b. Kinetix
- [x] (2026-05-17) Clone + `uv pip install -e ~/sim/kinetix` (with `jax` extra) — automated via `scripts/setup_sim.sh`
- [x] (2026-05-17) Smoke: `import kinetix` succeeds — submodules `editor / environment / models / render / util` reachable, JAX 0.9.0 on CPU
- [ ] `KinetixEnv.step_chunk()` runs at 50 Hz
- [ ] `inject_latency_ms` actually delays inference (used to stress RTC)

### 2c. MuJoCo
- [x] (2026-05-17) `import mujoco` works after `uv sync --extra sim` — mujoco 3.8.0, gymnasium 1.3.0; 10-step sim of a minimal MJCF (sphere falling under gravity) confirms the runtime
- [ ] First MJCF scene loaded (`tabletop_pick`) — needs an MJCF asset, not yet written
- [ ] `MuJoCoEnv.step_chunk()` runs to completion

### 2d. Open X-Embodiment
- [~] (2026-05-17) `tensorflow_datasets` + `rlds` installed
  - sub: tfds 4.9.10 + tensorflow 2.21.0 installed via the new `oxe` extra. `rlds` skipped — no macOS arm64 wheels, no sdist; tfds reads OXE episodes natively without it
- [x] (2026-05-17) Stream one episode from `fractal20220817_data` over GCS — anonymous access to `gs://gresearch/robotics`; first step has `image`, `natural_language_instruction`, `action`, `reward`
- [ ] `pi_stack.data.oxe.load_oxe()` yields `OXEEpisode` with embodiment tag
- [ ] `pi_stack.data.oxe.load_held_out()` yields exclusively held-out embodiments

## 3. Implementation — paper by paper

### 3a. π₀ (Oct 2024) — `pi_stack.models.pi0`
- [x] (2026-05-18) Backbone loader returns `(model, processor)` for PaliGemma 3B
  - sub: `PaliGemmaAdapter` wraps HF `PaliGemmaForConditionalGeneration` into the `(logits, features) + encode_image_features` contract `Pi0Policy` / `Pi07Policy` already expect; `load_backbone(PALIGEMMA_3B, device=..., dtype=...)` returns a ready-to-use adapter
  - sub: `Pi0Policy.from_pretrained(PALIGEMMA_3B, device='cuda')` is the one-liner factory
  - sub: 8 mocked unit tests in `tests/test_pi0_paligemma.py` verify the adapter shape contract; 1 env-gated integration test (`PI_STACK_RUN_REAL_PALIGEMMA=1`) for the real download
  - sub: end-to-end demo in `notebooks/08_pi0_paligemma_baseline.ipynb`; `scripts/inference_smoketest.py` is the pod-side sanity script
- [x] (2026-05-17) State encoder + late-fusion of state, image, language — `Pi0Policy.encode_context` concatenates VLM features with a state token; image patches prepended inside `TinyBackbone`
- [x] (2026-05-17) Action expert wired with cross-attention to VLM activations — `ActionExpert` is a `TransformerDecoder` whose memory is the (projected) VLM context
- [x] (2026-05-17) `Pi0Policy.predict_chunk(obs, language)` returns `(B, H, action_dim)` floats — verified in `tests/test_pi0.py::test_pi0_predict_chunk_shape`
- [x] (2026-05-17) Dummy-input forward pass < 200 ms on RTX 4090 (or MPS smoke) — TinyBackbone path runs in <100 ms end-to-end on CPU; whole `tests/test_pi0.py` finishes in <1 s

### 3b. FAST (Jan 2025) — `pi_stack.tokenization.fast`
- [x] (2026-05-17) `FASTTokenizer.encode(actions)` → `list[list[int]]` (B chunks of tokens; one list per batch element)
- [x] (2026-05-17) `FASTTokenizer.decode(tokens, H, D)` → `(B, H, D)` float32 array
- [x] (2026-05-17) Round-trip MSE < 1e-3 on a sanity trajectory — unit-amplitude phase-shifted sine, MSE consistently < 1e-3 in `tests/test_fast_tokenizer.py`; manual EDA in `notebooks/01_fast_eda.ipynb`
- [ ] Local DCT+BPE fallback (`use_official_tokenizer=False`)
  - sub: explicitly raises NotImplementedError. Deferred — the HF processor is the recommended path and works fine; we'd only need the local fallback to ablate the tokenizer choice

### 3c. Hi Robot (Feb 2025) — `pi_stack.models.hi_robot` + `pi_stack.data.synthetic`
- [ ] `segment_demonstration()` calls a VLM and parses JSON output
- [ ] `HiRobotPolicy.replan()` produces atomic subtask strings
- [ ] `step()` drives the low-level VLA via the produced subtask
- [ ] User-interjection re-plan window verified on a recorded demo

### 3d. π₀.₅ (Apr 2025) — `pi_stack.models.pi05`
- [x] (2026-05-17) Subtask language head autoregressive over FAST-style tokens — `Pi05Policy.subtask_head` is a `Linear(hidden, vocab)` over backbone features; `predict_subtask_logits()` returns next-token logits over the (configurable) subtask vocabulary
- [x] (2026-05-17) Co-training loss mixer respects `cotrain.*` weights from config — `Pi05Policy.cotrain_loss()` normalizes across present heads; verified across present/missing/empty key cases in `tests/test_pi05.py`
- [~] (2026-05-17) Subtask predictions feed back into action expert as conditioning
  - sub: subtask conditioning flows in via the language token pathway (caller appends the subtask to `language_ids`). A dedicated subtask-embedding context channel is a possible future refinement

### 3e. Knowledge Insulation (May 2025) — `pi_stack.training.ki`
- [x] (2026-05-17) Stop-gradient at VLM ↔ expert interface verified by autograd test — `insulate()` primitive + `tests/test_ki.py::test_end_to_end_ki_blocks_expert_gradient_from_vlm` runs a full toy forward and confirms VLM params receive zero gradient
- [x] (2026-05-17) Dual-loss step: discrete (FAST tokens, CE) + continuous (flow matching, MSE) — `ki_loss()` + `KITrainer.step()` reference recipe; toy run in `notebooks/02_ki_recipe.ipynb` shows backbone still drifts from the discrete loss while expert receives gradient
- [ ] Convergence speed measurably faster than diffusion-only baseline
  - blocker: needs a real backbone + dataset; revisit when we wire π₀ or π₀.₇
- [ ] VLM language-grounding metric does not degrade across training
  - blocker: same — needs real VLM and a held-out language probe

### 3f. Real-Time Chunking (Jun 2025) — `pi_stack.inference.rtc`
- [x] (2026-05-17) `flow_matching.euler_sample()` supports `prefix=` for inpainting — pins the first `P` actions after every Euler step. Verified end-to-end through `Pi0Policy.predict_chunk(..., prefix=...)` in `tests/test_pi0.py::test_pi0_rtc_inpainting_prefix_held_fixed` and demo'd in `notebooks/07_pi07_assembly.ipynb` §7
- [x] (2026-05-17) `RTCRunner` Algorithm 1 implemented (single inflight worker) — `pi_stack.inference.rtc.RTCRunner`; 7 unit tests in `tests/test_rtc.py` cover bootstrap, async overlap, fallback, prefix forwarding, and worker exception surfacing
- [~] (2026-05-17) Survives 350 ms injected latency on `kinetix_throw` without misses
  - sub: 200 ms-latency synthetic policy verified in `notebooks/03_rtc_loop.ipynb` (no swap blocking with H=50, overlap=10 at 50 Hz). The kinetix_throw integration is gated on wrapper-impl in §2b
- [~] (2026-05-17) Smooth motion preserved across chunk boundaries (jerk metric)
  - sub: qualitatively verified via inter-step Δt plot in the RTC notebook. A formal jerk metric will come with real env integration

### 3g. π*₀.₆ (Nov 2025) — `pi_stack.models.pi06` + `pi_stack.training.recap`
- [ ] Backbone swapped to Gemma 3 4B
  - blocker: depends on §3a (π₀ backbone wiring); π*₀.₆ inherits from Pi06Policy which is still a stub
- [x] (2026-05-17) Distributional value head (51 bins) + advantage estimator — `DistributionalValueHead` with C51 Bellman projection in `pi_stack.training.recap`; learns cluster-specific return distributions in `notebooks/06_recap_value_head.ipynb`
- [x] (2026-05-17) Advantage tokens injected as conditioning — `AdvantageConditioner` bucketizes continuous advantages and embeds them as a token; `top_bucket_token()` is the deployment-time "ask for above-average behavior" hook
- [x] (2026-05-17) `RECAPTrainer` alternates value and policy updates without collapse — verified end-to-end in `tests/test_recap.py::test_recap_trainer_step` and demonstrated in the notebook

### 3h. Human-to-Robot (Dec 2025) — `pi_stack.data.human_to_robot`
- [ ] Egocentric video loader (Ego4D or equivalent) emits `(frames, language)`
- [ ] Action-mask path so action loss is skipped on action-free clips
- [ ] Co-training run shows positive transfer on a robot held-out task

### 3i. MEM (Mar 2026) — `pi_stack.memory.mem`
- [x] (2026-05-17) Short-term ring buffer at configured fps — `ShortTermVideoMemory` downsamples to `short_term_fps`, evicts FIFO; verified across two `test_mem.py` cases
- [x] (2026-05-17) LLM summarizer turns finished subtasks into language summaries — pluggable `SummarizerFn` interface with a deterministic default; production swaps in Anthropic/HF model via constructor arg
- [x] (2026-05-17) `recall(query)` returns top-K summaries by relevance — brute-force cosine over a deterministic hash embedder (swap to `sentence-transformers` for real semantic recall, documented in `notebooks/04_mem_eda.ipynb`)
- [ ] Serialize/deserialize for episode resumption
  - sub: not implemented; the dataclasses are pickle-able as a fallback but a proper JSON/numpy round-trip would be cleaner
- [~] (2026-05-17) Integration test: 15-minute task with stateful question ("did I add salt?")
  - sub: 15-subtask kitchen scenario demonstrated in `notebooks/04_mem_eda.ipynb`; `recall("did I add salt yet?")` correctly retrieves the salt subtask. A formal `tests/` integration test is still nice-to-have

### 3j. RLT (Mar 2026) — `pi_stack.rlt.rl_token`
- [ ] RL token extracted from a chosen VLA layer
  - sub: the head takes the RL token as input (decoupled by design); the VLA-side hook lands when we wire π₀.₇
- [x] (2026-05-17) Actor MLP produces action residual on top of frozen VLA — reparameterized squashed-Gaussian actor with `action_residual_scale` clamp (default ±5%)
- [x] (2026-05-17) Critic + SAC-style update rule — twin Q nets, target networks with Polyak averaging, entropy auto-tuning. `tests/test_rlt.py::test_critic_can_fit_constant_target` shows convergence on a known target
- [~] (2026-05-17) Insertion task: 20% → ≥50% success after a few hours of practice
  - sub: SAC update mechanics verified on a synthetic insertion task in `notebooks/05_rlt_actor_critic.ipynb`; real-robot insertion is gated on env integration

### 3k. π₀.₇ (Apr 2026) — `pi_stack.models.pi07`
- [x] (2026-05-17) Subgoal-image encoder feeding context tokens — `Pi07Policy._encode_subgoal_images` re-uses the backbone's patch projection (SigLIP swap-in documented in `notebooks/07_pi07_assembly.ipynb`); accepts `(B, K, C, H, W)` and prepends `K * num_patches` tokens to the context
- [x] (2026-05-17) Episode-metadata token embedding (speed / quality / persona) — three `Embedding` tables (3 buckets each by default); `tests/test_pi07.py::test_pi07_metadata_token_changes_action` confirms the context shifts with the metadata
- [x] (2026-05-17) MEM hooked into the policy — `Pi07Policy(memory=mem)` + `recall_memory_tokens(query)` returns memory tokens projected into the policy's hidden space; `encode_context(..., memory_tokens=...)` prepends them
- [ ] Compositional out-of-the-box test (novel appliance) succeeds ≥1×
  - blocker: requires real VLA + real env; deferred

## 4. Training

- [ ] First π₀ training run on a small mixed dataset (Libero only)
- [ ] KI recipe verified end-to-end on the same run
- [ ] FAST tokenizer integrated in the data pipeline (no per-step bottleneck)
- [ ] Co-training mixer for π₀.₅ runs without OOMs
- [ ] RECAP RL run on π*₀.₆ (offline-first, then online)
- [ ] RLT online RL fine-tune wall-clock budget < 8 robot-hours
- [ ] Checkpointing + resume from `checkpoints/`
- [ ] wandb / tensorboard dashboards live for at least one run

## 5. Evaluation

- [ ] Libero-Spatial baseline numbers logged
- [ ] Libero-Object / -Goal / -10 / -90 numbers logged
- [ ] Kinetix throw + catch + balance numbers logged
- [ ] Kinetix latency stress test (350 ms) shows RTC graceful degradation
- [ ] MuJoCo tabletop baseline logged
- [ ] OXE held-out embodiment replay metrics logged
- [ ] Eval reports written to `outputs/<run-id>/eval.json`

## 6. Inference & deployment

- [ ] WebSocket server (`scripts/infer_server.py`) accepts msgpack frames
- [ ] Round-trip latency measured over LAN
- [ ] RTC streams chunks as they finish (not just whole-chunk responses)
- [ ] Real-robot dry run (or Libero-as-stand-in) over LAN
- [ ] 10–15 minute long-horizon task completes end-to-end with MEM enabled

## 8. RunPod GPU cluster wiring

The cluster is where the `NotImplementedError`-gated code paths actually
run: real PaliGemma / Gemma 3 forward passes, multi-GPU pre-training,
LoRA / KI fine-tunes, RLT online runs, large-scale evals. Local stays the
edit-and-pytest loop. See [`docs/runpod.md`](./docs/runpod.md) for the
step-by-step walkthrough; this section is the checkbox plan.

### 8a. Account + access
- [ ] Sign up at https://www.runpod.io ; verify billing
- [ ] Generate a RunPod **API key** (Settings → API Keys); store in your local password manager
- [ ] Generate an SSH key on the laptop if you don't already have one (`ssh-keygen -t ed25519`) and add the public key to RunPod (Settings → SSH Public Keys)
- [ ] (optional) Install the RunPod CLI: `pip install runpod` — useful for scripted pod start/stop

### 8b. Persistent storage
- [ ] Create a **Network Volume** (1-2 TB, in the region you'll be running pods)
  - sub: stores datasets, checkpoints, HF cache. Mounted at the same path on every pod (`/workspace`)
- [ ] Decide on a fixed directory layout for the volume:
  - `/workspace/pi-stack/` — git checkout (one per environment if branching)
  - `/workspace/data/` — Libero datasets, OXE subset, custom demos
  - `/workspace/checkpoints/` — training outputs (training script writes here)
  - `/workspace/hf-cache/` — `HF_HOME=/workspace/hf-cache` so weights download once
  - `/workspace/wandb/` — run logs if using wandb

### 8c. Container image
- [ ] Pick a base image — RunPod's official `runpod/pytorch:2.4.0-py3.11-cuda12.4-devel` (or newest equivalent) is a good starting point
- [ ] Author a thin `Dockerfile` on top that bakes in:
  - `uv` (so cold-start sync is fast)
  - System deps: `git`, `git-lfs`, `build-essential`, `libgl1` (for mujoco rendering), `ffmpeg` (for video logging)
  - A pre-populated `uv.lock` resolution (run `uv sync --extra all` during image build)
- [ ] Push to a registry — ghcr.io or Docker Hub — and reference it when creating pod templates

### 8d. Pod templates
- [ ] Create a **Dev pod template**: 1× A100 80GB PCIe (or H100 PCIe for ~2× speed), Network Volume mounted at `/workspace`, your custom image, JupyterLab + SSH enabled
- [ ] Create a **Training pod template**: 8× H100 SXM (NVLink), same volume + image. Spot pricing if you can tolerate interruptions; on-demand otherwise
- [ ] (optional) Create an **Inference pod template**: 1× A6000 48GB, lower-cost serve target if you don't need fine-tuning capacity

### 8e. First-pod bootstrap
- [ ] Start the dev pod
- [ ] SSH in: `ssh root@<pod-ip> -p <pod-port>`
- [ ] On the pod: `cd /workspace && git clone <your-fork-url> pi-stack && cd pi-stack`
- [ ] `uv sync --extra ml --extra dev --extra sim --extra jax --extra oxe`
- [ ] `bash scripts/setup_sim.sh` (the same idempotent script we use locally)
- [ ] `uv run pytest -q` — should be 65/65 green, same as local
- [ ] `export HF_HOME=/workspace/hf-cache` (persist across pod restarts via volume)
- [ ] `export HF_TOKEN=<token>` and `huggingface-cli whoami`

### 8f. Real-backbone smoke test
- [x] (2026-05-18) Wire `load_backbone(PALIGEMMA_3B)` — `PaliGemmaAdapter` in `pi_stack.models.backbones`; `Pi0Policy.from_pretrained(PALIGEMMA_3B)` factory
- [ ] Wire `load_backbone(GEMMA3_4B)` — same pattern, needs a `Gemma3Adapter` (mostly the same code minus the vision tower since Gemma 3 is text-only). Defer until we actually need π*₀.₆ / π₀.₇ on real weights
- [ ] First real load on the pod — `uv run python scripts/inference_smoketest.py` downloads ~6 GB into `/workspace/hf-cache/`; subsequent pod restarts use the cache
- [ ] Build `Pi07Policy(Pi07Config(backbone=GEMMA3_4B), backbone=load_backbone(GEMMA3_4B))` and run `predict_chunk(...)` on a real-sized input (224×224 image, 50-step horizon)
- [ ] Time the forward pass; record VRAM peak. Targets: < 200 ms forward on H100, < 30 GB VRAM at bf16
  - sub: `scripts/inference_smoketest.py` reports both numbers automatically

### 8g. Dataset hydration
- [ ] Pull Libero from the source clone on the volume; verify `from libero.libero import benchmark` works (re-run `scripts/setup_sim.sh` if needed)
- [ ] Download an OXE subset onto `/workspace/data/oxe/` — `tfds.load("fractal20220817_data", data_dir="/workspace/data/oxe")`
- [ ] Stage any custom teleop demos under `/workspace/data/demos/`
- [ ] Add `/workspace/data/` and `/workspace/checkpoints/` to `.gitignore` paths the training scripts respect (already true in the local repo)

### 8h. Multi-GPU training entrypoint
- [ ] Decide on launcher: `accelerate launch` is the lowest-friction option for FSDP/DDP across our PyTorch trainers
- [ ] Add `configs/runtime/single_gpu.yaml` and `configs/runtime/multi_gpu_ddp.yaml` (or use `accelerate config` to author them)
- [ ] Plumb the launcher through `scripts/train.py` so the same `--config configs/pi07.yaml` works on 1 GPU and 8 GPUs
- [ ] Verify scaling: a 100-step throughput-only run on 1 GPU vs 8 GPU shows ≥ 6.5× speedup (target 80%+ scaling efficiency)

### 8i. Code & data sync workflow
- [ ] Establish the loop: edit local → `git push` → on pod `git pull && uv sync` → run
- [ ] Or: use `rsync -av` for fast local-edit iteration (skip git when prototyping)
- [ ] Keep `.env` per-machine — never commit; put `HF_TOKEN`, `WANDB_API_KEY` there on both ends
- [ ] Set up wandb if you want shared run telemetry: `wandb login` on the pod with the same key as local

### 8j. Cost discipline
- [ ] Default: stop the dev pod whenever you're not actively using it (RunPod bills by the minute on running pods)
- [ ] Use **spot** pods for training runs that can checkpoint and resume — typically 30-50% cheaper than on-demand
- [ ] Set an idle-timeout (RunPod offers auto-shutdown) on the dev pod
- [ ] Add a `runpod-bill-check` cron / weekly habit to spot run-away pods
- [ ] Estimate budget per workflow ahead of time (e.g., π₀.₇ pre-train on 8× H100 SXM for 72 h ≈ $1.8-2.5 k)

### 8k. Production readiness (later)
- [ ] Containerize the inference server (`scripts/infer_server.py`) for a dedicated inference pod
- [ ] Configure secrets management — RunPod environment variables for `HF_TOKEN`, `WANDB_API_KEY`, robot LAN credentials
- [ ] (optional) Set up a one-click pod-from-template script in `scripts/runpod_launch.py` using the RunPod Python SDK

---

## 7. Cross-cutting

- [ ] `ruff check src tests` clean
- [ ] `mypy src` clean (or explicit `# type: ignore` with justification)
- [x] (2026-05-17) All paper modules pass `tests/test_smoke.py` — 6/6 green
- [ ] README quick-start works on a clean machine
- [ ] Paper → module map in `CLAUDE.md` matches the actual tree
- [ ] Each completed paper has a `docs/NN-shortname.md` design note
- [ ] Memory files in `~/.claude/projects/-Users-crohan-Documents-swat-cefi-PI-Stuff/memory/` reflect current decisions

---

## How to update this file

When a task here is finished:

1. Change `- [ ]` → `- [x] (YYYY-MM-DD)` for that line.
2. If the task is partial, change to `- [~] (YYYY-MM-DD)` and add a sub-bullet
   noting what's left.
3. If blocked, change to `- [!]` and add a sub-bullet with the blocker.
4. Stage and commit the checklist update **in the same commit** that closes
   the work. Don't let the file drift.

This rule is mirrored in `CLAUDE.md` — Claude is required to follow it before
reporting a task as complete.
