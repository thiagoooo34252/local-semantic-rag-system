"""Ingesta determinista de documentos locales en una colección ChromaDB."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, cast
from uuid import uuid4

import tiktoken
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from settings import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_DATA_PATH,
    DEFAULT_PERSIST_PATH,
    get_embedding_model,
    load_project_environment,
    require_openai_api_key,
)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOKEN_ENCODING = "cl100k_base"
SEPARATORS = ["\n\n", "\n", " ", ""]
MANIFEST_NAME = ".ingestion-manifest.json"
SUPPORTED_SUFFIXES = {".md", ".txt"}


@dataclass(frozen=True)
class IngestionResult:
    """Resumen serializable de una ejecución de ingestión."""

    status: Literal["indexed", "skipped"]
    source_documents: int
    chunks: int
    fingerprint: str


@lru_cache(maxsize=1)
def _get_encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding(TOKEN_ENCODING)


def count_tokens(text: str) -> int:
    """Cuenta tokens con `cl100k_base` sin interpretar tokens especiales."""
    return len(_get_encoding().encode(text, disallowed_special=()))


def clean_text(text: str) -> str:
    """Normaliza saltos y espacios horizontales sin destruir párrafos."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[^\S\n]+", " ", normalized)
    normalized = "\n".join(line.strip() for line in normalized.splitlines())
    return re.sub(r"\n{3,}", "\n\n", normalized).strip()


def load_source_documents(data_path: Path) -> list[Document]:
    """Carga archivos directos `.md` y `.txt` en orden determinista."""
    if not data_path.exists():
        raise FileNotFoundError(f"No existe el directorio de datos: {data_path}")
    if not data_path.is_dir():
        raise ValueError(f"La ruta de datos no es un directorio: {data_path}")

    paths = sorted(
        (
            path
            for path in data_path.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        ),
        key=lambda path: path.name,
    )
    if not paths:
        raise ValueError(
            f"No hay archivos .md o .txt directos para ingerir en {data_path}."
        )

    documents: list[Document] = []
    for path in paths:
        try:
            content = clean_text(path.read_text(encoding="utf-8"))
        except UnicodeDecodeError as error:
            raise ValueError(
                f"El archivo no está codificado en UTF-8: {path.name}"
            ) from error
        if content:
            documents.append(
                Document(page_content=content, metadata={"source": path.name})
            )

    if not documents:
        raise ValueError("Los archivos de datos están vacíos después de la limpieza.")
    return documents


def deterministic_chunk_id(source: str, chunk_index: int, content_hash: str) -> str:
    """Genera un identificador estable a partir de fuente, posición y contenido."""
    identity = f"{source}\0{chunk_index}\0{content_hash}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _split_text_safely(
    splitter: RecursiveCharacterTextSplitter, text: str
) -> list[str]:
    chunks: list[str] = []
    for candidate in splitter.split_text(text):
        if count_tokens(candidate) <= CHUNK_SIZE:
            chunks.append(candidate)
            continue

        safety_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE - 1,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=count_tokens,
            separators=SEPARATORS,
            keep_separator="start",
            strip_whitespace=True,
        )
        safe_candidates = safety_splitter.split_text(candidate)
        if any(count_tokens(chunk) > CHUNK_SIZE for chunk in safe_candidates):
            raise RuntimeError("No fue posible cumplir el límite seguro de tokens.")
        chunks.extend(safe_candidates)
    return chunks


def split_documents(
    documents: Sequence[Document], embedding_model: str
) -> tuple[list[Document], list[str]]:
    """Divide documentos en fragmentos de hasta 500 tokens con solape de 50."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=count_tokens,
        separators=SEPARATORS,
        keep_separator="start",
        strip_whitespace=True,
    )
    chunks: list[Document] = []
    chunk_ids: list[str] = []

    for document in documents:
        source = str(document.metadata["source"])
        source_chunks = _split_text_safely(splitter, document.page_content)
        for chunk_index, content in enumerate(source_chunks):
            token_total = count_tokens(content)
            if token_total > CHUNK_SIZE:
                raise RuntimeError(
                    f"El fragmento {chunk_index} de {source} "
                    f"supera {CHUNK_SIZE} tokens."
                )
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            chunk_id = deterministic_chunk_id(source, chunk_index, content_hash)
            metadata: dict[str, str | int] = {
                "source": source,
                "chunk_index": chunk_index,
                "content_sha256": content_hash,
                "embedding_model": embedding_model,
                "chunk_id": chunk_id,
            }
            chunks.append(Document(page_content=content, metadata=metadata))
            chunk_ids.append(chunk_id)

    if not chunks:
        raise ValueError("La división no produjo fragmentos indexables.")
    return chunks, chunk_ids


def _manifest_payload(
    documents: Sequence[Document], collection_name: str, embedding_model: str
) -> dict[str, Any]:
    sources = [
        {
            "source": str(document.metadata["source"]),
            "sha256": hashlib.sha256(document.page_content.encode("utf-8")).hexdigest(),
        }
        for document in documents
    ]
    configuration = {
        "version": 1,
        "sources": sources,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "separators": SEPARATORS,
        "token_encoding": TOKEN_ENCODING,
        "collection_name": collection_name,
        "embedding_model": embedding_model,
    }
    serialized = json.dumps(
        configuration, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return {
        **configuration,
        "fingerprint": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
    }


def _read_manifest(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return cast(dict[str, Any], value) if isinstance(value, dict) else None


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _new_vector_store(
    persist_path: Path,
    collection_name: str,
    embeddings: Embeddings,
) -> Chroma:
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_path),
        collection_metadata={"hnsw:space": "cosine"},
    )


def _collection_ids(vector_store: Chroma) -> list[str]:
    result = vector_store.get(include=[])
    return [str(item) for item in result.get("ids", [])]


def ingest_documents(
    *,
    data_path: Path = DEFAULT_DATA_PATH,
    persist_path: Path = DEFAULT_PERSIST_PATH,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    force: bool = False,
    embeddings: Embeddings | None = None,
    embedding_model: str | None = None,
) -> IngestionResult:
    """Indexa el dataset o evita trabajo si su huella no cambió."""
    selected_model = embedding_model or get_embedding_model()
    documents = load_source_documents(data_path)
    manifest = _manifest_payload(documents, collection_name, selected_model)
    fingerprint = str(manifest["fingerprint"])

    if persist_path.exists() and not persist_path.is_dir():
        raise ValueError(f"La ruta de persistencia no es un directorio: {persist_path}")
    persist_path.mkdir(parents=True, exist_ok=True)
    manifest_path = persist_path / MANIFEST_NAME

    if embeddings is None:
        require_openai_api_key()
        embeddings = OpenAIEmbeddings(model=selected_model)
    vector_store = _new_vector_store(persist_path, collection_name, embeddings)
    existing_ids = _collection_ids(vector_store)
    previous_manifest = _read_manifest(manifest_path)

    if (
        not force
        and existing_ids
        and previous_manifest is not None
        and previous_manifest.get("fingerprint") == fingerprint
    ):
        return IngestionResult(
            status="skipped",
            source_documents=len(documents),
            chunks=len(existing_ids),
            fingerprint=fingerprint,
        )

    chunks, chunk_ids = split_documents(documents, selected_model)
    transaction_id = uuid4().hex
    replacement_name = f"rebuild-{transaction_id}"
    backup_name = f"backup-{transaction_id}"
    previous_renamed = False
    replacement_promoted = False

    replacement_store = _new_vector_store(persist_path, replacement_name, embeddings)
    try:
        # Build under an isolated name so embedding failures cannot touch live data.
        replacement_store.add_documents(chunks, ids=chunk_ids)

        vector_store._collection.modify(name=backup_name)
        previous_renamed = True
        replacement_store._collection.modify(name=collection_name)
        replacement_promoted = True
        _write_manifest(manifest_path, manifest)
    except Exception:
        if replacement_promoted:
            replacement_store._collection.modify(name=replacement_name)
        if previous_renamed:
            vector_store._collection.modify(name=collection_name)
        replacement_store.delete_collection()
        raise

    vector_store.delete_collection()
    return IngestionResult(
        status="indexed",
        source_documents=len(documents),
        chunks=len(chunks),
        fingerprint=fingerprint,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ingiere documentos locales en una colección persistente de ChromaDB."
        )
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--persist-path", type=Path, default=DEFAULT_PERSIST_PATH)
    parser.add_argument("--collection-name", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Ejecuta la interfaz de línea de comandos de ingestión."""
    load_project_environment()
    arguments = _parse_args()
    result = ingest_documents(
        data_path=arguments.data_path,
        persist_path=arguments.persist_path,
        collection_name=arguments.collection_name,
        force=arguments.force,
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
