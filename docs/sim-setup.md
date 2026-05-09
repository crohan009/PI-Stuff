# Simulation & Evaluation setup

The four pieces of the eval stack and how to install each.

| Piece | What it is | When to use |
|---|---|---|
| **Libero** | Standard VLA benchmark suite (5 sub-suites) | General π₀-class performance regression |
| **Kinetix** | Dynamic / force-based 2D physics tasks | Validating Real-Time Chunking (RTC) latency robustness |
| **MuJoCo** | Tabletop manipulation baseline | Sanity-check policies before Libero |
| **OXE** | 22+ embodiment teleop dataset | Cross-embodiment held-out replay evaluation |

All four are gated behind the `sim` extra so the bare install stays light:

```bash
uv sync --extra sim
```

That gives you `mujoco` and `gymnasium`. Libero, Kinetix, and OXE are
installed from source — instructions below.

---

## Libero

Repo: https://github.com/Lifelong-Robot-Learning/LIBERO

Not on PyPI. Install from source into the project venv:

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git ~/sim/libero
cd ~/sim/libero
uv pip install -e .
# native deps: robosuite + mujoco — both come with the install
```

Sanity check:

```bash
uv run python -c "from libero.libero import benchmark; print(benchmark.get_benchmark_dict())"
```

Five sub-suites: `libero_spatial`, `libero_object`, `libero_goal`,
`libero_10`, `libero_90`. Use `configs/eval/libero.yaml` and override
`suite:` per run.

---

## Kinetix

Repo: https://github.com/FlairOx/Kinetix

JAX-based, so install the `jax` extra alongside:

```bash
uv sync --extra sim --extra jax
git clone https://github.com/FlairOx/Kinetix.git ~/sim/kinetix
cd ~/sim/kinetix
uv pip install -e .
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

Dataset, not a simulator. Stored as RLDS / TFDS. Pulls TensorFlow:

```bash
uv pip install tensorflow-cpu tensorflow-datasets rlds
```

Datasets live on Google Cloud Storage. To stream a single embodiment:

```python
import tensorflow_datasets as tfds
ds = tfds.load("fractal20220817_data", data_dir="gs://gresearch/robotics")
```

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
