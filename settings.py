"""Configuración compartida del sistema RAG local."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_DATA_PATH = Path("data")
DEFAULT_PERSIST_PATH = Path("vectorstore")
DEFAULT_COLLECTION_NAME = "local_semantic_rag"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_TOP_K = 4
SUPPORTED_REASONING_EFFORTS = ("none", "low", "medium", "high", "xhigh", "max")


def load_project_environment() -> None:
    """Carga `.env` salvo que las pruebas deshabiliten dotenv explícitamente."""
    if os.getenv("PYTHON_DOTENV_DISABLED") != "1":
        load_dotenv()


def get_embedding_model() -> str:
    """Devuelve el modelo de embeddings configurado."""
    return os.getenv("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL).strip()


def get_chat_model() -> str:
    """Devuelve el modelo de chat configurado."""
    return os.getenv("OPENAI_CHAT_MODEL", DEFAULT_CHAT_MODEL).strip()


def get_reasoning_effort() -> str | None:
    """Devuelve y valida el esfuerzo de razonamiento opcional."""
    reasoning_effort = os.getenv("OPENAI_REASONING_EFFORT", "").strip()
    if not reasoning_effort:
        return None
    if reasoning_effort not in SUPPORTED_REASONING_EFFORTS:
        supported_values = ", ".join(SUPPORTED_REASONING_EFFORTS)
        raise ValueError(
            f"OPENAI_REASONING_EFFORT debe ser uno de: {supported_values}."
        )
    return reasoning_effort


def get_persist_path() -> Path:
    """Devuelve la ruta configurada para la base vectorial."""
    return Path(os.getenv("RAG_PERSIST_PATH", str(DEFAULT_PERSIST_PATH)))


def get_collection_name() -> str:
    """Devuelve el nombre configurado para la colección."""
    return os.getenv("RAG_COLLECTION_NAME", DEFAULT_COLLECTION_NAME).strip()


def get_top_k() -> int:
    """Devuelve la cantidad configurada de fragmentos por consulta."""
    raw_value = os.getenv("RAG_TOP_K", str(DEFAULT_TOP_K))
    try:
        return int(raw_value)
    except ValueError as error:
        raise ValueError("RAG_TOP_K debe ser un número entero entre 3 y 5.") from error


def require_openai_api_key() -> None:
    """Verifica que exista una clave sin exponer su valor."""
    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise RuntimeError(
            "Falta OPENAI_API_KEY. Copie .env.example a .env y configure su clave."
        )
