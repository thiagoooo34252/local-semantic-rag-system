from __future__ import annotations

import json
import sys

import pytest

import ingest
import rag
from ingest import IngestionResult
from models import RagResponse


def test_ingestion_cli_prints_spanish_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    result = IngestionResult(
        status="skipped",
        source_documents=4,
        chunks=11,
        fingerprint="abc123",
    )
    monkeypatch.setattr(ingest, "ingest_documents", lambda **_kwargs: result)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ingest.py",
            "--data-path",
            "datos",
            "--persist-path",
            "base",
            "--collection-name",
            "coleccion",
            "--force",
        ],
    )

    ingest.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "status": "skipped",
        "source_documents": 4,
        "chunks": 11,
        "fingerprint": "abc123",
    }


def test_rag_cli_runs_async_and_prints_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def fake_response(query: str) -> RagResponse:
        assert query == "consulta de prueba"
        return RagResponse(answer="Respuesta local.", references=["fuente.md"])

    monkeypatch.setattr(rag, "get_rag_response", fake_response)
    monkeypatch.setattr(sys, "argv", ["rag.py", "consulta de prueba"])

    rag.main()

    assert json.loads(capsys.readouterr().out) == {
        "answer": "Respuesta local.",
        "references": ["fuente.md"],
    }
