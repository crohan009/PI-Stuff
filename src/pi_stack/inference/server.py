"""WebSocket inference server.

A workstation runs the model; the robot (or simulator) is a thin client over
LAN. Keeps inference on the GPU box and avoids shipping weights to the robot.

Wire protocol (msgpack frames):
    request:  {"obs": <obs>, "language": <str>, "request_id": <int>}
    response: {"actions": <H x D float32>, "request_id": <int>}

TODO:
  - implement asyncio + websockets server
  - hook RTCRunner so chunks are streamed as they finish, not after the whole
    chunk is computed (see RTC paper for streaming semantics)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8765
    max_message_bytes: int = 16 * 1024 * 1024


def serve(config: ServerConfig | None = None) -> None:
    raise NotImplementedError
