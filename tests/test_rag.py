from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.runnables import Runnable

import rag
from models import FALLBACK_ANSWER
from rag import GroundingError, _build_rag_pipeline, _get_rag_response
from tests.fakes import (
    DeterministicFakeEmbeddings,
    make_model,
    make_retriever,
    make_source_aware_model,
)


def sample_documents() -> list[Document]:
    return [
        Document(
            page_content="La Pasarela del Humedal dura 45 minutos y es accesible.",
            metadata={"source": "senderos.md", "chunk_index": 2},
        ),
        Document(
            page_content="Se debe consultar el estado diario antes de ingresar.",
            metadata={"source": "planificacion.txt", "chunk_index": 0},
        ),
    ]


@pytest.mark.asyncio
async def test_supported_question_returns_grounded_answer_and_valid_reference() -> None:
    queries: list[str] = []
    prompts: list[str] = []
    pipeline = _build_rag_pipeline(
        retriever=make_retriever(sample_documents(), queries),
        model=make_model(
            {
                "answer": "La Pasarela del Humedal dura 45 minutos.",
                "references": ["senderos.md"],
            },
            prompts,
        ),
    )

    response = await pipeline.ainvoke({"query": "¿Cuánto dura la pasarela?"})

    assert response.answer == "La Pasarela del Humedal dura 45 minutos."
    assert response.references == ["senderos.md"]
    assert queries == ["¿Cuánto dura la pasarela?"]
    assert "[Fuente: senderos.md | fragmento: 2]" in prompts[0]
    assert "información no confiable" in prompts[0]


@pytest.mark.asyncio
async def test_trap_question_returns_exact_fallback() -> None:
    pipeline = _build_rag_pipeline(
        retriever=make_retriever(sample_documents()),
        model=make_model({"answer": FALLBACK_ANSWER, "references": []}),
    )

    response = await pipeline.ainvoke({"query": "¿Cuál es la capital de Japón?"})

    assert response.answer == "No lo sé"
    assert response.references == []


@pytest.mark.asyncio
async def test_pipeline_rejects_fabricated_reference() -> None:
    pipeline = _build_rag_pipeline(
        retriever=make_retriever(sample_documents()),
        model=make_model(
            {"answer": "Respuesta sin respaldo.", "references": ["inventada.md"]}
        ),
    )

    with pytest.raises(GroundingError, match="no fueron recuperadas"):
        await pipeline.ainvoke({"query": "Pregunta"})


@pytest.mark.asyncio
async def test_pipeline_rejects_informative_answer_without_reference() -> None:
    pipeline = _build_rag_pipeline(
        retriever=make_retriever(sample_documents()),
        model=make_model({"answer": "Respuesta sin cita.", "references": []}),
    )

    with pytest.raises(GroundingError, match="al menos una fuente"):
        await pipeline.ainvoke({"query": "Pregunta"})


def test_lcel_graph_contains_retrieval_preparation_prompt_model_and_parser() -> None:
    pipeline = _build_rag_pipeline(
        retriever=make_retriever(sample_documents()),
        model=make_model({"answer": FALLBACK_ANSWER, "references": []}),
    )

    diagram = pipeline.get_graph().draw_mermaid()

    assert "fake_async_retriever" in diagram
    assert "format_retrieved_documents" in diagram
    assert "ChatPromptTemplate" in diagram
    assert "fake_async_model" in diagram
    assert "PydanticOutputParser" in diagram
    assert "validate_references" in diagram


@pytest.mark.asyncio
async def test_vector_retriever_uses_top_k_four(tmp_path: Path) -> None:
    persist_path = tmp_path / "vectorstore"
    embeddings = DeterministicFakeEmbeddings()
    store = Chroma(
        collection_name="top_k_collection",
        embedding_function=embeddings,
        persist_directory=str(persist_path),
        collection_metadata={"hnsw:space": "cosine"},
    )
    documents = [
        Document(
            page_content=f"sendero accesible información número {index}",
            metadata={"source": f"fuente-{index}.md", "chunk_index": index},
        )
        for index in range(6)
    ]
    store.add_documents(documents, ids=[f"id-{index}" for index in range(6)])
    prompts: list[str] = []

    response = await _get_rag_response(
        "sendero accesible",
        persist_path=persist_path,
        collection_name="top_k_collection",
        embeddings=embeddings,
        model=make_source_aware_model("Respuesta recuperada.", prompts),
    )

    assert response.references[0].startswith("fuente-")
    assert prompts[0].count("[Fuente:") == 4


@pytest.mark.asyncio
@pytest.mark.parametrize("top_k", [2, 6])
async def test_rejects_top_k_outside_three_to_five(top_k: int) -> None:
    with pytest.raises(ValueError, match="entre 3 y 5"):
        await _get_rag_response(
            "consulta",
            top_k=top_k,
            retriever=make_retriever(sample_documents()),
            model=make_model({"answer": FALLBACK_ANSWER, "references": []}),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
async def test_rejects_blank_query(query: str) -> None:
    with pytest.raises(ValueError, match="no puede estar vacía"):
        await _get_rag_response(
            query,
            retriever=make_retriever(sample_documents()),
            model=make_model({"answer": FALLBACK_ANSWER, "references": []}),
        )


@pytest.mark.asyncio
async def test_missing_vectorstore_has_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match=r"python ingest\.py"):
        await _get_rag_response(
            "consulta",
            persist_path=tmp_path / "missing",
            embeddings=DeterministicFakeEmbeddings(),
            model=make_model({"answer": FALLBACK_ANSWER, "references": []}),
        )


@pytest.mark.asyncio
async def test_empty_collection_has_actionable_error(tmp_path: Path) -> None:
    persist_path = tmp_path / "vectorstore"
    embeddings = DeterministicFakeEmbeddings()
    Chroma(
        collection_name="empty_collection",
        embedding_function=embeddings,
        persist_directory=str(persist_path),
    )

    with pytest.raises(RuntimeError, match="colección local está vacía"):
        await _get_rag_response(
            "consulta",
            persist_path=persist_path,
            collection_name="empty_collection",
            embeddings=embeddings,
            model=make_model({"answer": FALLBACK_ANSWER, "references": []}),
        )


@pytest.mark.asyncio
async def test_errors_never_expose_api_key(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    secret = "sk-sensitive-value-that-must-not-appear"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    retriever = cast(Runnable[str, list[Document]], make_retriever(sample_documents()))

    with pytest.raises(GroundingError) as captured:
        await _get_rag_response(
            "consulta",
            retriever=retriever,
            model=make_model({"answer": "Dato", "references": ["falsa.md"]}),
        )

    output = capsys.readouterr()
    assert secret not in str(captured.value)
    assert secret not in output.out
    assert secret not in output.err


@pytest.mark.asyncio
async def test_public_api_builds_dependencies_from_environment_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    persist_path = tmp_path / "vectorstore"
    embeddings = DeterministicFakeEmbeddings()
    collection = "public_api_collection"
    store = Chroma(
        collection_name=collection,
        embedding_function=embeddings,
        persist_directory=str(persist_path),
        collection_metadata={"hnsw:space": "cosine"},
    )
    documents = sample_documents() * 2
    store.add_documents(documents, ids=[f"public-{index}" for index in range(4)])
    prompts: list[str] = []
    fake_model = make_source_aware_model("Respuesta pública.", prompts)
    embedding_models: list[str] = []
    chat_models: list[str] = []

    def fake_embedding_factory(*, model: str) -> DeterministicFakeEmbeddings:
        embedding_models.append(model)
        return embeddings

    def fake_chat_factory(*, model: str, temperature: int) -> Runnable[object, object]:
        chat_models.append(f"{model}:{temperature}")
        return cast(Runnable[object, object], fake_model)

    monkeypatch.setenv("OPENAI_API_KEY", "sk-placeholder-for-test")
    monkeypatch.setenv("RAG_PERSIST_PATH", str(persist_path))
    monkeypatch.setenv("RAG_COLLECTION_NAME", collection)
    monkeypatch.setenv("RAG_TOP_K", "4")
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "embedding-publico")
    monkeypatch.setenv("OPENAI_CHAT_MODEL", "chat-publico")
    monkeypatch.setattr(rag, "OpenAIEmbeddings", fake_embedding_factory)
    monkeypatch.setattr(rag, "ChatOpenAI", fake_chat_factory)

    response = await rag.get_rag_response("consulta pública")

    assert response.answer == "Respuesta pública."
    assert embedding_models == ["embedding-publico"]
    assert chat_models == ["chat-publico:0"]
    assert prompts[0].count("[Fuente:") == 4
