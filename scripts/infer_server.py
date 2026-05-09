"""WebSocket inference server entrypoint.

Usage::

    uv run python scripts/infer_server.py --config configs/rtc.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pi_stack.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the pi-stack inference server.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    server_cfg = config.get("server", {})
    print(f"[serve] host={server_cfg.get('host', '0.0.0.0')} port={server_cfg.get('port', 8765)}")
    raise NotImplementedError("RTC streaming server not implemented yet.")


if __name__ == "__main__":
    main()
