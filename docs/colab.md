# Running notebook 00 on Google Colab

The fastest way to exercise the **real PaliGemma 3B baseline** without setting
up a RunPod pod. Colab's free T4 has 16 GB VRAM — PaliGemma 3B fits in
~6 GB at bf16 with room to spare.

> Just want the link? **[Open notebook 00 in Colab](https://colab.research.google.com/github/crohan009/PI-Stuff/blob/main/notebooks/00_pi0_paligemma_baseline.ipynb)**

## One-time setup (per Google account)

1. **Create the HF_TOKEN secret.** In any Colab notebook, click the 🔑 key
   icon in the left sidebar (or **Tools → User secrets**). Add a secret
   named exactly `HF_TOKEN` with your HuggingFace token (read access is
   enough). Toggle the **Notebook access** switch on. This is reused
   across notebooks, so you do it once.

2. **Accept the PaliGemma license** on HuggingFace if you haven't:
   https://huggingface.co/google/paligemma-3b-pt-224 → click "Access
   request" → "Authorize". Takes a minute; required because PaliGemma is
   gated.

## Per-session run

1. **[Open the notebook](https://colab.research.google.com/github/crohan009/PI-Stuff/blob/main/notebooks/00_pi0_paligemma_baseline.ipynb)** — Colab opens it
   directly from the GitHub raw URL.
2. **Runtime → Change runtime type → T4 GPU** (free tier). If you have
   Colab Pro: L4 or A100 are faster but T4 is enough for PaliGemma 3B.
3. **Run the first two cells** — the markdown setup notes and the
   bootstrap code. The bootstrap:
   - Clones https://github.com/crohan009/PI-Stuff into `/content/pi-stack`
   - `pip install -e /content/pi-stack[ml]` — installs transformers,
     sentencepiece, scipy, sklearn. **Does NOT touch torch** (Colab's
     CUDA torch stays).
   - Loads `HF_TOKEN` from the secret you set up above.
   - First cold run takes ~30 seconds.
4. **Run the rest of the cells.** The notebook walks you through:
   - Processor smoke test
   - Full PaliGemma load (~6 GB download on first run; cached at
     `~/.cache/huggingface/` between cells but **not** between Colab
     sessions — every fresh runtime re-downloads)
   - One Pi0 forward pass + flow-matching action chunk sample with timing
   - One KI training step on the real backbone with grad-norm sanity check

Expected wall-clock on T4:
- Bootstrap (cell 2): ~30 s
- Processor load (cell 4): ~5 s
- Full PaliGemma load (cell 6): ~90 s first time (download + load), ~10 s subsequent
- Forward + chunk sample: ~500 ms
- KI step: ~2 s

## Pointing at a fork or branch

The bootstrap reads two optional env vars before cloning:

```python
import os
os.environ['PI_STACK_REPO']   = 'https://github.com/<your-fork>/PI-Stuff.git'
os.environ['PI_STACK_BRANCH'] = 'experimental-branch'
# then re-run the bootstrap cell
```

Set those in a cell *before* the bootstrap if you want a non-default source.

## Common gotchas

| Symptom | Cause | Fix |
|---|---|---|
| `ImportError: No module named pi_stack` after bootstrap | Kernel cached `sys.path` before the install | Bootstrap cell adds `/content/pi-stack/src` to `sys.path`; if still broken, **Runtime → Restart session** then re-run from the top |
| `OSError: Could not find HF_TOKEN` | Secret not set, or not toggled on for this notebook | Tools → User secrets → confirm `HF_TOKEN` exists + notebook access is on |
| `403 Forbidden` from HF | License not accepted | Visit the PaliGemma model page on HF, click "Authorize" |
| `OutOfMemoryError` on cell 6 | Not on GPU runtime, or stale state | Runtime → Change runtime type → GPU; Runtime → Restart session |
| Bootstrap clones but the install hangs | Colab disk full (rare) | Runtime → Disconnect and delete runtime → restart |

## Why not use Colab as the day-to-day dev env?

- **Sessions are ephemeral.** Every disconnect re-downloads PaliGemma
  (6 GB). RunPod's Network Volume persists weights across pod restarts.
- **Free tier has hard limits** (~12 h continuous, GPU availability not
  guaranteed). Fine for one-off notebook runs, painful for training.
- **No persistent SSH / IDE integration.** Edit-and-pytest loop is local;
  Colab is just for running this one notebook on a real GPU.

For sustained work, see [`docs/runpod.md`](./runpod.md) — the dev-pod
plan is engineered for this. Colab notebook 00 exists for *fast
verification* that the PaliGemma baseline works end-to-end.
