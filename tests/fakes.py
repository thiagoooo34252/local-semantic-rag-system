from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import Runnable, RunnableLambda


class DeterministicFakeEmbeddings(Embeddings):
    """Embeddings locales por bolsa de palabras para pruebas reproducibles."""

    dimensions = 128

    def __init__(self) -> None:
        self.document_batches = 0
        self.query_calls = 0

    @classmethod
    def _embed(cls, text: str) -> list[float]:
        vector = [0.0] * cls.dimensions
        tokens = re.findall(r"\w+", text.casefold(), flags=re.UNICODE)
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:2]) % cls.dimensions
            sign = 1.0 if digest[2] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_batches += 1
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls += 1
        return self._embed(text)


def make_retriever(
    documents: Sequence[Document], queries: list[str] | None = None
) -> Runnable[str, list[Document]]:
    async def retrieve(query: str) -> list[Document]:
        if queries is not None:
            queries.append(query)
        return list(documents)

    return RunnableLambda(retrieve, name="fake_async_retriever")


def make_model(
    payload: dict[str, object], prompts: list[str] | None = None
) -> Runnable[Any, Any]:
    async def respond(prompt: Any) -> str:
        rendered = prompt.to_string()
        if prompts is not None:
            prompts.append(rendered)
        return json.dumps(payload, ensure_ascii=False)

    return RunnableLambda(respond, name="fake_async_model")


def make_source_aware_model(answer: str, prompts: list[str]) -> Runnable[Any, Any]:
    async def respond(prompt: Any) -> str:
        rendered = prompt.to_string()
        prompts.append(rendered)
        match = re.search(r"Fuentes permitidas: (\[[^\n]*\])", rendered)
        sources = json.loads(match.group(1)) if match else []
        return json.dumps(
            {"answer": answer, "references": sources[:1]}, ensure_ascii=False
        )

    return RunnableLambda(respond, name="source_aware_async_model")
