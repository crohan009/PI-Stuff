"""Evaluation entrypoint.

Dispatches across simulators (Libero / Kinetix / MuJoCo) and the OXE replay
harness based on the eval config's ``suite`` field.

Usage::

    uv run python scripts/eval.py --model configs/pi0.yaml --eval configs/eval/libero.yaml
    uv run python scripts/eval.py --model configs/pi07.yaml --eval configs/eval/kinetix.yaml
    uv run python scripts/eval.py --model configs/pi05.yaml --eval configs/eval/oxe.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pi_stack.utils.config import load_config


def _is_oxe(eval_cfg: dict) -> bool:
    return eval_cfg.get("mode") == "replay" and "held_out_embodiments" in eval_cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a pi-stack policy.")
    parser.add_argument("--model", type=Path, required=True, help="Model config (configs/*.yaml).")
    parser.add_argument("--eval", type=Path, required=True, help="Eval config (configs/eval/*.yaml).")
    parser.add_argument("--episodes", type=int, default=None, help="Override episodes from config.")
    args = parser.parse_args()

    model_cfg = load_config(args.model)
    eval_cfg = load_config(args.eval)

    model_name = model_cfg.get("model", {}).get("name", "<unset>")
    print(f"[eval] model={model_name}")

    if _is_oxe(eval_cfg):
        print(f"[eval] OXE replay over {len(eval_cfg.get('held_out_embodiments', []))} held-out embodiments")
        # TODO: dispatch to OXE replay harness — see pi_stack.data.oxe.load_held_out
        raise NotImplementedError("OXE replay eval not implemented yet.")

    suite = eval_cfg["suite"]
    episodes = args.episodes or eval_cfg.get("episodes", 20)
    print(f"[eval] suite={suite} episodes={episodes}")

    # Avoid importing heavy native deps until we actually need them.
    from pi_stack.envs import make_env

    env = make_env(suite)
    print(f"[eval] env={env.suite} embodiment={env.embodiment} control_hz={env.control_hz}")
    # TODO: load policy from `model_cfg`, run `episodes` rollouts via env.step_chunk,
    # aggregate success rates per task, write a JSON report to ./outputs/.
    raise NotImplementedError("Eval rollout loop not implemented yet.")


if __name__ == "__main__":
    main()
