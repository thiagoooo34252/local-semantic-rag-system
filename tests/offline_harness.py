from __future__ import annotations

import asyncio
import json
import socket
import tempfile
from pathlib import Path
from typing import Any, cast

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnableLambda

from ingest import ingest_documents
from models import FALLBACK_ANSWER
from rag import _build_rag_pipeline
from tests.fakes import DeterministicFakeEmbeddings

ROOT = Path(__file__).resolve().parents[1]
COLLECTION = "offline_harness"
MODEL = "deterministic-fake-embedding"


async def run_harness() -> dict[str, object]:
    """Ejecuta ingestión, reapertura, recuperación y respuestas sin red."""
    network_attempts: list[str] = []
    original_connect = socket.socket.connect

    def blocked_connect(_socket: socket.socket, address: object) -> None:
        network_attempts.append(str(address))
        raise AssertionError("El harness detectó un intento de red.")

    socket.socket.connect = blocked_connect
    try:
        with tempfile.TemporaryDirectory() as temporary_directory:
            persist_path = Path(temporary_directory) / "vectorstore"
            ingestion = ingest_documents(
                data_path=ROOT / "data",
                persist_path=persist_path,
                collection_name=COLLECTION,
                embeddings=DeterministicFakeEmbeddings(),
                embedding_model=MODEL,
            )
            recreated = Chroma(
                collection_name=COLLECTION,
                embedding_function=DeterministicFakeEmbeddings(),
                persist_directory=str(persist_path),
                collection_metadata={"hnsw:space": "cosine"},
            )
            retriever = cast(
                Runnable[str, list[Document]],
                recreated.as_retriever(search_kwargs={"k": 4}),
            )
            retrieved = await retriever.ainvoke(
                "¿Qué recorrido es accesible y cuánto tiempo dura?"
            )
            retrieved_sources = {
                str(document.metadata["source"]) for document in retrieved
            }
            assert "02_senderos_y_accesibilidad.txt" in retrieved_sources

            async def respond(prompt: Any) -> str:
                rendered = prompt.to_string()
                if "capital de Japón" in rendered:
                    payload = {"answer": FALLBACK_ANSWER, "references": []}
                else:
                    payload = {
                        "answer": (
                            "La Pasarela del Humedal es accesible y dura "
                            "aproximadamente 45 minutos."
                        ),
                        "references": ["02_senderos_y_accesibilidad.txt"],
                    }
                return json.dumps(payload, ensure_ascii=False)

            pipeline = _build_rag_pipeline(
                retriever=retriever,
                model=RunnableLambda(respond, name="offline_fake_async_model"),
            )
            supported = await pipeline.ainvoke(
                {"query": "¿Qué recorrido es accesible y cuánto tiempo dura?"}
            )
            trap = await pipeline.ainvoke({"query": "¿Cuál es la capital de Japón?"})
            assert supported.references == ["02_senderos_y_accesibilidad.txt"]
            assert trap.answer == FALLBACK_ANSWER and trap.references == []
            return {
                "ingestion_status": ingestion.status,
                "chunks": ingestion.chunks,
                "recreated_collection_count": len(recreated.get(include=[])["ids"]),
                "retrieved_sources": sorted(retrieved_sources),
                "supported": supported.model_dump(),
                "trap": trap.model_dump(),
                "network_attempts": network_attempts,
            }
    finally:
        socket.socket.connect = original_connect


if __name__ == "__main__":
    print(json.dumps(asyncio.run(run_harness()), ensure_ascii=False, indent=2))
