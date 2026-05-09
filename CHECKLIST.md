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

- [ ] Install `uv` (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- [ ] `uv sync` succeeds (bare install)
- [ ] `uv sync --extra ml` succeeds (transformers + accelerate + scipy)
- [ ] `uv sync --extra dev` and `uv run pytest -q` passes smoke tests
- [ ] `HF_TOKEN` set in `.env` and `huggingface-cli whoami` works
- [ ] GPU torch wheels installed for the local CUDA version (or MPS/CPU fallback verified on macOS)
- [ ] `uv run python -c "from transformers import AutoProcessor; AutoProcessor.from_pretrained('physical-intelligence/fast', trust_remote_code=True)"` succeeds
- [ ] (optional) `wandb login` if using `track` extra

## 2. Simulation & evaluation stack

### 2a. Libero
- [ ] Clone + `uv pip install -e ~/sim/libero`
- [ ] Smoke: `from libero.libero import benchmark` imports
- [ ] `pi_stack.envs.libero.LiberoEnv.reset()` returns a real obs dict
- [ ] `step_chunk()` runs an H=50 chunk through robosuite
- [ ] All 5 sub-suites instantiable (Spatial / Object / Goal / 10 / 90)

### 2b. Kinetix
- [ ] Clone + `uv pip install -e ~/sim/kinetix` (with `jax` extra)
- [ ] Smoke: `import kinetix` succeeds
- [ ] `KinetixEnv.step_chunk()` runs at 50 Hz
- [ ] `inject_latency_ms` actually delays inference (used to stress RTC)

### 2c. MuJoCo
- [ ] `import mujoco` works after `uv sync --extra sim`
- [ ] First MJCF scene loaded (`tabletop_pick`)
- [ ] `MuJoCoEnv.step_chunk()` runs to completion

### 2d. Open X-Embodiment
- [ ] `tensorflow_datasets` + `rlds` installed
- [ ] Stream one episode from `fractal20220817_data` over GCS
- [ ] `pi_stack.data.oxe.load_oxe()` yields `OXEEpisode` with embodiment tag
- [ ] `pi_stack.data.oxe.load_held_out()` yields exclusively held-out embodiments

## 3. Implementation — paper by paper

### 3a. π₀ (Oct 2024) — `pi_stack.models.pi0`
- [ ] Backbone loader returns `(model, processor)` for PaliGemma 3B
- [ ] State encoder + late-fusion fusion of state, image, language
- [ ] Action expert wired with cross-attention to VLM activations
- [ ] `Pi0Policy.predict_chunk(obs, language)` returns `(H, action_dim)` floats
- [ ] Dummy-input forward pass < 200 ms on RTX 4090 (or MPS smoke)

### 3b. FAST (Jan 2025) — `pi_stack.tokenization.fast`
- [ ] `FASTTokenizer.encode(actions)` → `list[int]`
- [ ] `FASTTokenizer.decode(tokens, H, D)` → `(H, D)` float array
- [ ] Round-trip MSE < 1e-3 on a sanity trajectory
- [ ] Local DCT+BPE fallback (`use_official_tokenizer=False`)

### 3c. Hi Robot (Feb 2025) — `pi_stack.models.hi_robot` + `pi_stack.data.synthetic`
- [ ] `segment_demonstration()` calls a VLM and parses JSON output
- [ ] `HiRobotPolicy.replan()` produces atomic subtask strings
- [ ] `step()` drives the low-level VLA via the produced subtask
- [ ] User-interjection re-plan window verified on a recorded demo

### 3d. π₀.₅ (Apr 2025) — `pi_stack.models.pi05`
- [ ] Subtask language head autoregressive over FAST-style tokens
- [ ] Co-training loss mixer respects `cotrain.*` weights from config
- [ ] Subtask predictions feed back into action expert as conditioning

### 3e. Knowledge Insulation (May 2025) — `pi_stack.training.ki`
- [ ] Stop-gradient at VLM ↔ expert interface verified by autograd test
- [ ] Dual-loss step: discrete (FAST tokens) + continuous (flow matching)
- [ ] Convergence speed measurably faster than diffusion-only baseline
- [ ] VLM language-grounding metric does not degrade across training

### 3f. Real-Time Chunking (Jun 2025) — `pi_stack.inference.rtc`
- [ ] `flow_matching.euler_sample()` supports `inpaint_prefix=`
- [ ] `RTCRunner` Algorithm 1 implemented (single inflight worker)
- [ ] Survives 350 ms injected latency on `kinetix_throw` without misses
- [ ] Smooth motion preserved across chunk boundaries (jerk metric)

### 3g. π*₀.₆ (Nov 2025) — `pi_stack.models.pi06` + `pi_stack.training.recap`
- [ ] Backbone swapped to Gemma 3 4B
- [ ] Distributional value head (51 bins) + advantage estimator
- [ ] Advantage tokens injected as conditioning
- [ ] `RECAPTrainer` alternates value and policy updates without collapse

### 3h. Human-to-Robot (Dec 2025) — `pi_stack.data.human_to_robot`
- [ ] Egocentric video loader (Ego4D or equivalent) emits `(frames, language)`
- [ ] Action-mask path so action loss is skipped on action-free clips
- [ ] Co-training run shows positive transfer on a robot held-out task

### 3i. MEM (Mar 2026) — `pi_stack.memory.mem`
- [ ] Short-term ring buffer at configured fps
- [ ] LLM summarizer turns finished subtasks into language summaries
- [ ] `recall(query)` returns top-K summaries by relevance
- [ ] Serialize/deserialize for episode resumption
- [ ] Integration test: 15-minute task with stateful question ("did I add salt?")

### 3j. RLT (Mar 2026) — `pi_stack.rlt.rl_token`
- [ ] RL token extracted from a chosen VLA layer
- [ ] Actor MLP produces action residual on top of frozen VLA
- [ ] Critic + SAC-style update rule
- [ ] Insertion task: 20% → ≥50% success after a few hours of practice

### 3k. π₀.₇ (Apr 2026) — `pi_stack.models.pi07`
- [ ] Subgoal-image encoder (SigLIP) feeding context tokens
- [ ] Episode-metadata token embedding (speed / quality / persona)
- [ ] MEM hooked into the policy
- [ ] Compositional out-of-the-box test (novel appliance) succeeds ≥1×

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

## 7. Cross-cutting

- [ ] `ruff check src tests` clean
- [ ] `mypy src` clean (or explicit `# type: ignore` with justification)
- [ ] All paper modules pass `tests/test_smoke.py`
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
