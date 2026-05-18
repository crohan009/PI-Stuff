"""Pi0 + PaliGemma inference smoketest.

Standalone sanity check for the experimental baseline. Loads the real
PaliGemma 3B backbone, builds Pi0Policy on top, runs one forward pass +
flow-matching chunk sample, and prints latency / memory stats.

Use this to verify a fresh pod is correctly set up before kicking off
longer training/eval. Expected runtime: under 30 s on a single H100;
30 s-2 min on MPS depending on memory.

Usage::

    uv run python scripts/inference_smoketest.py
    uv run python scripts/inference_smoketest.py --device cpu       # very slow
    uv run python scripts/inference_smoketest.py --device cuda --dtype bfloat16
    uv run python scripts/inference_smoketest.py --skip-load        # adapter API only
"""

from __future__ import annotations

import argparse
import time


def _pick_device(arg: str | None) -> str:
    import torch

    if arg is not None:
        return arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _pick_dtype(name: str | None, device: str):
    import torch

    if name is not None:
        return getattr(torch, name)
    # bf16 on CUDA + MPS; fp32 on CPU (bf16 is slow on x86 CPU).
    return torch.bfloat16 if device in ("cuda", "mps") else torch.float32


def main() -> None:
    parser = argparse.ArgumentParser(description="Pi0+PaliGemma smoketest")
    parser.add_argument("--device", default=None, help="cuda | mps | cpu")
    parser.add_argument("--dtype", default=None, help="bfloat16 | float16 | float32")
    parser.add_argument("--skip-load", action="store_true",
                        help="Skip the real model load — useful to verify the script itself")
    parser.add_argument("--prompt", default="pick up the red cube",
                        help="Language instruction for the forward pass")
    args = parser.parse_args()

    if args.skip_load:
        print("[smoketest] --skip-load set; not loading PaliGemma. Exiting cleanly.")
        return

    import torch
    from PIL import Image

    from pi_stack.models.backbones import PALIGEMMA_3B
    from pi_stack.models.pi0 import Pi0Config, Pi0Policy

    device = _pick_device(args.device)
    dtype = _pick_dtype(args.dtype, device)
    print(f"[smoketest] device={device} dtype={dtype}")
    print(f"[smoketest] loading {PALIGEMMA_3B.hf_repo} (~6 GB bf16; first call also downloads weights)")

    t0 = time.perf_counter()
    policy = Pi0Policy.from_pretrained(
        PALIGEMMA_3B,
        config=Pi0Config(backbone=PALIGEMMA_3B, state_dim=14, image_resolution=224),
        device=device,
        dtype=dtype,
    )
    t_load = time.perf_counter() - t0
    print(f"[smoketest] load time: {t_load:.1f}s")
    if device == "cuda":
        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"[smoketest] CUDA peak memory after load: {peak_gb:.2f} GB")

    # Build inputs via the processor — PaliGemma needs the special
    # <image>...<bos> token layout.
    processor = policy.backbone.processor
    image = Image.new("RGB", (224, 224), color=(128, 128, 128))
    inputs = processor(text=args.prompt, images=image, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    pixel_values = inputs["pixel_values"].to(device).to(dtype)
    state = torch.zeros(1, 14, device=device, dtype=dtype)

    # Forward + sample.
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    with torch.no_grad():
        chunk = policy.predict_chunk(
            images=pixel_values,
            state=state,
            language_ids=input_ids,
        )
    if device == "cuda":
        torch.cuda.synchronize()
    t_infer = time.perf_counter() - t1

    print(f"[smoketest] chunk shape: {tuple(chunk.shape)} dtype={chunk.dtype}")
    print(f"[smoketest] forward + flow-matching sample: {t_infer*1000:.1f} ms")
    print(f"[smoketest] action range: [{chunk.float().min().item():+.3f}, {chunk.float().max().item():+.3f}]")
    print("[smoketest] ok")


if __name__ == "__main__":
    main()
