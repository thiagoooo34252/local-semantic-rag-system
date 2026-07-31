from __future__ import annotations

import os
import socket
from collections.abc import Iterator

import pytest

os.environ["PYTHON_DOTENV_DISABLED"] = "1"
os.environ["ANONYMIZED_TELEMETRY"] = "FALSE"
os.environ.pop("OPENAI_API_KEY", None)


@pytest.fixture(autouse=True)
def block_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    def blocked_connection(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Las pruebas no permiten conexiones de red.")

    monkeypatch.setattr(socket.socket, "connect", blocked_connection)
    monkeypatch.setattr(socket, "create_connection", blocked_connection)
    yield
