# Simulation & Evaluation setup

The four pieces of the eval stack and how to install each.

| Piece | What it is | When to use |
|---|---|---|
| **Libero** | Standard VLA benchmark suite (5 sub-suites) | General π₀-class performance regression |
| **Kinetix** | Dynamic / force-based 2D physics tasks | Validating Real-Time Chunking (RTC) latency robustness |
| **MuJoCo** | Tabletop manipulation baseline | Sanity-check policies before Libero |
| **OXE** | 22+ embodiment teleop dataset | Cross-embodiment held-out replay evaluation |

The MuJoCo + Gymnasium piece lives in the `sim` extra. OXE (TensorFlow +
TFDS) lives in a separate `oxe` extra to keep TF opt-in:

```bash
uv sync --extra ml --extra sim --extra jax --extra oxe
```

Libero and Kinetix are installed from source (no PyPI wheels). After
`uv sync` you need to re-apply the two editable installs because
`uv sync` prunes anything it doesn't manage. A helper script handles
the whole flow:

```bash
bash scripts/setup_sim.sh           # clones to ~/sim by default
SIM_ROOT=/custom bash scripts/setup_sim.sh
```

The script is idempotent — re-run it after every `uv sync` to put the
Libero/Kinetix editables back in place.

---

## Libero

Repo: https://github.com/Lifelong-Robot-Learning/LIBERO

Not on PyPI. Use `scripts/setup_sim.sh` (recommended) or follow the
manual steps below.

**Two gotchas the script handles for you:**

1. **Namespace package init.** Libero's top-level `libero/` directory has
   no `__init__.py`. setuptools' `find_packages()` returns empty without
   it, which yields a no-op editable install. The script drops an empty
   `__init__.py` into the clone.
2. **Interactive first-run prompt.** `libero/libero/__init__.py` calls
   `input()` on first import asking about a dataset folder, which causes
   `EOFError` in non-interactive runs. The script pre-seeds
   `~/.libero/config.yaml` with defaults that mirror `get_default_path_dict()`.

Manual install (equivalent to what the script does):

```bash
git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git ~/sim/libero
touch ~/sim/libero/libero/__init__.py
mkdir -p ~/.libero
cat > ~/.libero/config.yaml <<YAML
benchmark_root: ~/sim/libero/libero/libero
bddl_files: ~/sim/libero/libero/libero/bddl_files
init_states: ~/sim/libero/libero/libero/init_files
datasets: ~/sim/libero/libero/libero/../datasets
assets: ~/sim/libero/libero/libero/assets
YAML
uv pip install -e ~/sim/libero
```

Sanity check:

```bash
uv run python -c "from libero.libero import benchmark; print(benchmark.get_benchmark_dict())"
```

Six entries appear at the `benchmark_dict` layer: `libero_spatial`,
`libero_object`, `libero_goal`, `libero_10`, `libero_90`, `libero_100`
(the 100-task union). Use `configs/eval/libero.yaml` and override
`suite:` per run.

**Note on robosuite.** Libero's `requirements.txt` pins robosuite 1.4,
old numpy, transformers 4.21, and the deprecated `gym` package — those
would clobber our stack if enforced. We deliberately install Libero
without its requirements (its `setup.py` has `install_requires=[]`, so
the editable install is a no-op for deps). To actually *run* Libero
tasks you'll need to resolve the robosuite dep separately — this is a
compatibility battle deferred until we implement
`pi_stack.envs.libero.LiberoEnv.reset()`.

---

## Kinetix

Repo: https://github.com/FlairOx/Kinetix

JAX-based with a proper `pyproject.toml` (unlike Libero). The setup
script handles it; manual install:

```bash
uv sync --extra sim --extra jax
git clone --depth 1 https://github.com/FlairOx/Kinetix.git ~/sim/kinetix
uv pip install -e ~/sim/kinetix
```

Why Kinetix specifically: the RTC paper uses it because dynamic tasks
(throwing, catching, balancing) punish any inference latency. Static
benchmarks like Libero won't surface latency bugs.

To stress-test RTC, set `inject_latency_ms: 350` in
`configs/eval/kinetix.yaml`.

---

## MuJoCo

Pure pip — comes with the `sim` extra:

```bash
uv sync --extra sim
uv run python -c "import mujoco; print(mujoco.__version__)"
```

We use raw `mujoco` 3.x (not `dm_control`) to keep the dep surface thin.
Custom MJCF scenes go in `data/mujoco_scenes/` (gitignored).

---

## Open X-Embodiment (OXE)

Dataset, not a simulator. Stored as RLDS / TFDS. Lives in the `oxe`
extra (kept separate from `ml` because TensorFlow is heavy):

```bash
uv sync --extra ml --extra oxe
```

**The `rlds` Python package is intentionally omitted on macOS arm64.**
It has no Apple Silicon wheels and no source distribution (last release
v0.1.8 is Linux x86_64 only). The good news: `tensorflow_datasets`
streams OXE episodes natively — `rlds` only adds episode-slicing helpers
that we don't currently need. If you later need them on Linux, add
`rlds` to your local extras manually.

Stream a single embodiment from the public GCS bucket:

```python
import tensorflow_datasets as tfds
ds = tfds.load(
    "fractal20220817_data",
    split="train[:1]",
    data_dir="gs://gresearch/robotics",
)
for episode in ds.take(1):
    for step in episode["steps"]:
        print(step["observation"]["natural_language_instruction"])
        break
```

You'll see a TF warning about missing Google credentials — public buckets
read anonymously, so the warning is informational and the data streams
either way.

For mass downloads (TB-scale), use `gsutil rsync gs://gresearch/robotics
~/sim/oxe`. Most users only need a handful of embodiments — see the
`DEFAULT_EMBODIMENTS` list in `src/pi_stack/data/oxe.py`.

OXE evaluation is **replay-based**: score the policy by comparing predicted
action chunks against ground-truth chunks on held-out embodiments. Config
in `configs/eval/oxe.yaml`.

---

## Quick sanity matrix

| Stack piece | Smoke command |
|---|---|
| Libero | `uv run python -c "from libero.libero import benchmark"` |
| Kinetix | `uv run python -c "import kinetix"` |
| MuJoCo | `uv run python -c "import mujoco"` |
| OXE | `uv run python -c "import tensorflow_datasets as tfds; tfds.builder('fractal20220817_data')"` |

If a stack piece fails its smoke command, **don't fight it** — record the
failure in `CHECKLIST.md` under the Setup section and move on. Most of the
arc can be developed against just MuJoCo + Libero.

## Why `uv sync` "forgets" Libero / Kinetix

`uv sync` enforces that the venv matches `pyproject.toml` + `uv.lock`
exactly. Anything installed via raw `uv pip install` outside of that
lockfile is treated as not-supposed-to-be-here and pruned on next sync.

Libero and Kinetix are git clones, not PyPI packages, so they live in
this "outside-of-lockfile" zone. The fix is to re-run
`scripts/setup_sim.sh` after every `uv sync` — it's idempotent (clones
are skipped if present, only the editable installs are reapplied),
and it takes about a second on the second-and-onwards runs.

If this gets annoying, the more invasive fix is to declare them in
`[tool.uv.sources]` with absolute paths, but that bakes machine-specific
paths into `pyproject.toml`. Not worth it for a single-developer repo.
