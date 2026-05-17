# CLAUDE.md — pi-stack

Project-specific guidance for Claude Code when working in this repo. Global
preferences live in `~/.claude/CLAUDE.md` and apply on top of this file.

## Master checklist — READ AND OBEY

The single source of truth for "what's done / what's next" is
[`CHECKLIST.md`](./CHECKLIST.md) at the repo root. It covers bootstrap,
local environment, simulation stack, paper-by-paper implementation,
training, evaluation, inference, and cross-cutting hygiene.

**Hard rule (non-negotiable):** whenever you complete work that maps to a
checklist item, before reporting back to the user you MUST:

1. Open `CHECKLIST.md`.
2. Flip the matching `- [ ]` to `- [x] (YYYY-MM-DD)` (use today's date in
   ISO format — `2026-05-10`, etc.). For partial work use
   `- [~] (YYYY-MM-DD)` with a sub-bullet describing what's left. For
   blocked items use `- [!]` with a sub-bullet noting the blocker.
3. Include the checklist edit in the **same** commit that closes the work,
   so the file never drifts from reality.
4. If you can't find a matching item, add one under the right section
   instead of skipping the update.

Do this even for small completions. The checklist is how the user
audits progress across sessions; if it's stale, the project is stale.

**What this rule is not:** it isn't a hook. There's no harness mechanism
that auto-flips checkboxes — the auto-update is *you*, every time, before
yielding the turn. Don't ask permission, don't summarize the rule, just
do it.

**Belt-and-suspenders:** `.claude/settings.json` defines a `Stop` hook that
prints a one-line `systemMessage` reminding the user (and you, on the next
turn) about the rule above. The hook is a nudge, not the enforcement. The
obligation is still this rule. If you see the reminder in the transcript,
that's why.

## What this repo is

`pi-stack` is a single-developer implementation of the **Physical Intelligence**
research arc from **π₀** (Oct 2024) through **π₀.₇** (Apr 2026). The arc is
summarized in [`About-PI.md`](./About-PI.md). All 11 PI papers are in
[`papers/`](./papers/) (gitignored, kept locally) along with foundational prior
work in `papers/cited/`.

The user is implementing this **single-handed**. Velocity matters more than
breadth. Prefer leveraging pre-trained foundation models and the official
[`openpi`](https://github.com/Physical-Intelligence/openpi) library over
reimplementing components from scratch.

## Paper → module map (canonical)

When adding code, place it in the module that matches the paper that
**introduced** the technique, even if a later paper reuses it. Cross-reference
via imports rather than duplicating logic.

| Paper | Primary module | Key dependencies |
|---|---|---|
| π₀ | `pi_stack.models.pi0` | PaliGemma, flow-matching action expert |
| FAST | `pi_stack.tokenization.fast` | `AutoProcessor.from_pretrained("physical-intelligence/fast")` |
| Hi Robot | `pi_stack.models.hi_robot` + `pi_stack.data.synthetic` | uses π₀ as System 1 |
| π₀.₅ | `pi_stack.models.pi05` | co-training mix (mobile, web, VQA) |
| KI | `pi_stack.training.ki` | stop-gradient between VLM and action expert |
| RTC | `pi_stack.inference.rtc` | async inpainting; wraps any flow-matching policy |
| π*₀.₆ | `pi_stack.models.pi06` + `pi_stack.training.recap` | advantage-conditioned RL |
| Human-to-Robot | `pi_stack.data.human_to_robot` | egocentric video → robot transfer |
| MEM | `pi_stack.memory.mem` | dual-memory (short-term video + long-term language) |
| RLT | `pi_stack.rlt.rl_token` | tiny actor-critic on a frozen VLA |
| π₀.₇ | `pi_stack.models.pi07` | Gemma 3 backbone, multi-modal context, MEM |

### Simulation & evaluation stack

| Stack piece | Module | Use case |
|---|---|---|
| Libero (5 sub-suites) | `pi_stack.envs.libero` + `configs/eval/libero.yaml` | Standard VLA benchmark |
| Kinetix | `pi_stack.envs.kinetix` + `configs/eval/kinetix.yaml` | Stress-test RTC on dynamic tasks |
| MuJoCo | `pi_stack.envs.mujoco` + `configs/eval/mujoco.yaml` | Tabletop baseline |
| Open X-Embodiment (OXE) | `pi_stack.data.oxe` + `configs/eval/oxe.yaml` | Cross-embodiment held-out replay |

OXE lives in `data/`, not `envs/`, because it's a dataset (replay-evaluated)
rather than a simulator. Common eval interface: `pi_stack.envs.base.BaseEvalEnv`
— concrete sims must satisfy this protocol, and `pi_stack.envs.make_env(suite)`
dispatches by name. Install steps for each piece are in
[`docs/sim-setup.md`](./docs/sim-setup.md).

## Conventions

### Code
- **Layout:** `src/pi_stack/`. Imports always go through the package
  (`from pi_stack.tokenization.fast import FASTTokenizer`), never relative
  paths from `scripts/` or notebooks.
- **Type hints:** required on public functions. Internal helpers can skip.
- **Docstrings:** every module top-level docstring must cite the paper it
  implements (filename in `papers/`) and, where helpful, the section number.
- **Comments:** only when the WHY is non-obvious. Don't restate the paper.

### Dependencies
- Use `uv add` / `uv sync`. Never `pip install` into the project venv.
- Heavy deps (torch, jax, mujoco, libero, kinetix) live in `[project.optional-dependencies]`
  extras (`ml`, `gpu`, `jax`, `sim`, `serve`, `track`, `dev`).
- The bare install (`uv sync` with no extras) must work on CPU-only macOS.

### Configs
- One YAML per model/recipe in `configs/`.
- Configs are loaded with `pi_stack.utils.config.load_config` (Pydantic).
- Don't bake hyperparameters into Python defaults — put them in the config.

### Data and checkpoints
- `data/` and `checkpoints/` are gitignored. Use them for local artifacts.
- For shared/public datasets (Libero, OXE), prefer streaming over downloading.

### Testing
- `pytest` in `tests/`. Smoke tests only at scaffold time; real tests
  arrive when the corresponding module gains real code.
- Don't write tests that require GPU or downloaded weights unless gated by
  a `pytest.mark.slow` / `pytest.mark.gpu` marker.

## When the user references a paper by name

Default to the matching module from the table above first. If the user says
"Hi Robot," look in `pi_stack.models.hi_robot` and `pi_stack.data.synthetic`
before grepping the whole tree.

## When the user references a simulator or dataset

- **"Libero" / "Libero-Spatial / -Object / -Goal / -10 / -90"** →
  `pi_stack.envs.libero`, config `configs/eval/libero.yaml`.
- **"Kinetix"** → `pi_stack.envs.kinetix`, config `configs/eval/kinetix.yaml`.
  Specifically used for **RTC** validation; static benchmarks won't surface
  latency bugs.
- **"MuJoCo"** → `pi_stack.envs.mujoco`, config `configs/eval/mujoco.yaml`.
- **"OXE" / "Open X-Embodiment"** → `pi_stack.data.oxe` (dataset, not sim),
  config `configs/eval/oxe.yaml`. Replay-based, tags every episode with its
  embodiment.

## Things to avoid

- **Don't reorganize the per-paper module layout.** It's the user's primary
  navigation aid. Add new files inside existing modules instead.
- **Don't invent fix recipes for things that aren't broken.** This is a
  research scaffold, not a production codebase — premature hardening adds
  friction without benefit.
- **Don't pull in another VLA framework** (lerobot, octo, openvla) without
  asking. The intent is to track the PI / openpi line specifically.
- **Don't auto-download paper PDFs.** They're already in `papers/`. The
  `scrape_pi.py` helper at the root exists for re-running the crawl if the
  user asks; it's not a normal workflow step.

## Two environments — local vs RunPod

This project deliberately splits work across two homes:

| Environment | What runs here | What does NOT run here |
|---|---|---|
| **Local laptop** (`~/Documents/swat/cefi/PI-Stuff/`) | Test suite, TinyBackbone policy code, MEM logic, RTC algorithm, FAST tokenizer download, KI / RLT / RECAP mechanics on toy data | Real PaliGemma / Gemma 3 forward passes, full pre-training, real Libero/Kinetix rollouts |
| **RunPod pod** (cloned from git on the pod) | Real backbones, KI fine-tuning, RECAP RL, RLT online runs, multi-GPU pre-training, real sim rollouts | The fast inner-loop edit-and-pytest cycle (do that local) |

**Code lives in git. Data and checkpoints live on a RunPod Network Volume.**
Same code path runs both places — what differs is the `backbone=` kwarg
passed to `Pi0Policy` / `Pi05Policy` / `Pi06Policy` / `Pi07Policy`. Local
uses `TinyBackbone`; RunPod uses `load_backbone(GEMMA3_4B)` (or PaliGemma).

### `NotImplementedError` paths — when they finally get exercised

A handful of code paths intentionally raise `NotImplementedError` because
they require hardware/weights we don't have locally:

- `pi_stack.models.backbones.load_backbone(spec)` for PaliGemma / Gemma 3 / SigLIP
- `pi_stack.tokenization.fast.FASTTokenizer` with `use_official_tokenizer=False`
  (the local DCT+BPE fallback — deferred; HF processor works on both envs)
- Real-env rollout paths in `pi_stack.envs.{libero,kinetix,mujoco}` —
  blocked on robosuite dep resolution (see `docs/sim-setup.md`)

Treat hitting one of these as a signal: either this work belongs on the
RunPod pod, or there's a deferred-task acknowledgement in `CHECKLIST.md`.
**Don't paper over them locally** — the gates are intentional.

### Deploying changes to the pod

The expected workflow:
1. Edit + test locally with TinyBackbone. `uv run pytest -q` is the inner loop.
2. Commit and push when green.
3. SSH into the pod, `git pull`, run the same command with the real config
   (e.g. `uv run python scripts/train.py --config configs/pi07.yaml`).

Step 3 is the only place the gated paths actually fire. The walkthrough is
in [`docs/runpod.md`](./docs/runpod.md); the wiring checklist is §8 of
[`CHECKLIST.md`](./CHECKLIST.md).

## External resources

- **openpi:** https://github.com/Physical-Intelligence/openpi (PI's official lib)
- **FAST tokenizer:** HF `physical-intelligence/fast`
- **PaliGemma:** HF `google/paligemma-3b-pt-224`
- **Gemma 3:** HF `google/gemma-3-4b-pt`
- **SigLIP:** HF `google/siglip-base-patch16-224`
- **Libero:** https://github.com/Lifelong-Robot-Learning/LIBERO
- **Kinetix:** https://github.com/FlairOx/Kinetix
- **RunPod:** https://www.runpod.io (cluster host); docs at https://docs.runpod.io
