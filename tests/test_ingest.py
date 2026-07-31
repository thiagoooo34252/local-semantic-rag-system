from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import ingest
from ingest import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    MANIFEST_NAME,
    _split_text_safely,
    clean_text,
    count_tokens,
    deterministic_chunk_id,
    ingest_documents,
    load_source_documents,
    split_documents,
)
from tests.fakes import DeterministicFakeEmbeddings

COLLECTION = "test_collection"
MODEL = "fake-embedding-v1"


class DeterministicByteEncoding:
    @staticmethod
    def encode(text: str, *, disallowed_special: object = "all") -> list[int]:
        return list(text.encode("utf-8"))


@pytest.fixture(autouse=True)
def use_deterministic_token_encoding(monkeypatch: pytest.MonkeyPatch) -> None:
    encoding = DeterministicByteEncoding()
    monkeypatch.setattr(ingest, "_get_encoding", lambda: encoding)


class FailingDocumentEmbeddings(DeterministicFakeEmbeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("simulated replacement failure")


def write_dataset(
    path: Path, content: str = "Contenido inicial de la reserva."
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "fuente.md").write_text(content, encoding="utf-8")


def read_ids(path: Path, embeddings: DeterministicFakeEmbeddings) -> list[str]:
    store = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(path),
    )
    return [str(value) for value in store.get(include=[])["ids"]]


def seed_previous_corpus(tmp_path: Path) -> tuple[Path, Path, str, str]:
    data_path = tmp_path / "data"
    persist_path = tmp_path / "vectorstore"
    previous_content = "Corpus anterior que debe seguir disponible."
    write_dataset(data_path, previous_content)
    ingest_documents(
        data_path=data_path,
        persist_path=persist_path,
        collection_name=COLLECTION,
        embeddings=DeterministicFakeEmbeddings(),
        embedding_model=MODEL,
    )
    previous_manifest = (persist_path / MANIFEST_NAME).read_text(encoding="utf-8")
    return data_path, persist_path, previous_content, previous_manifest


def assert_previous_state(
    persist_path: Path, previous_content: str, previous_manifest: str
) -> None:
    store = Chroma(
        collection_name=COLLECTION,
        embedding_function=DeterministicFakeEmbeddings(),
        persist_directory=str(persist_path),
    )
    matches = store.similarity_search("corpus anterior", k=1)

    assert matches[0].page_content == previous_content
    actual_manifest = (persist_path / MANIFEST_NAME).read_text(encoding="utf-8")
    assert actual_manifest == previous_manifest


def test_loads_direct_sources_sorted_and_cleans_without_losing_paragraphs(
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "data"
    data_path.mkdir()
    (data_path / "b.txt").write_text(
        " Segundo\r\n\r\n\r\n\t párrafo   final ", encoding="utf-8"
    )
    (data_path / "a.md").write_text("Primero", encoding="utf-8")
    (data_path / "ignorado.csv").write_text("No cargar", encoding="utf-8")
    nested = data_path / "nested"
    nested.mkdir()
    (nested / "oculto.md").write_text("No cargar", encoding="utf-8")

    documents = load_source_documents(data_path)

    assert [document.metadata["source"] for document in documents] == ["a.md", "b.txt"]
    assert documents[1].page_content == "Segundo\n\npárrafo final"


def test_clean_text_normalizes_horizontal_whitespace_and_blank_lines() -> None:
    assert clean_text(" uno\t dos\r\n\r\n\r\n\r\ntres ") == "uno dos\n\ntres"


@pytest.mark.parametrize("kind", ["missing", "empty", "blank"])
def test_rejects_missing_or_empty_dataset(tmp_path: Path, kind: str) -> None:
    data_path = tmp_path / kind
    if kind != "missing":
        data_path.mkdir()
    if kind == "blank":
        (data_path / "vacio.txt").write_text(" \n\n ", encoding="utf-8")

    with pytest.raises((FileNotFoundError, ValueError)):
        load_source_documents(data_path)


def test_token_chunking_uses_500_with_50_overlap_and_preserves_unicode() -> None:
    text = " ".join(f"palabra{i} café <|endoftext|>" for i in range(600))
    document = Document(page_content=text, metadata={"source": "unicode.md"})

    chunks, _ = split_documents([document], MODEL)

    token_totals = [count_tokens(chunk.page_content) for chunk in chunks]
    assert CHUNK_SIZE == 500
    assert CHUNK_OVERLAP == 50
    assert len(chunks) > 2
    assert max(token_totals) <= 500
    assert "café" in "".join(chunk.page_content for chunk in chunks)
    assert "<|endoftext|>" in "".join(chunk.page_content for chunk in chunks)


def test_safety_resplit_handles_token_recombination_overshoot() -> None:
    text = " ".join(f"palabra{index}" for index in range(1000))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=count_tokens,
        separators=["\n\n", "\n", " ", ""],
        keep_separator="start",
        strip_whitespace=True,
    )

    chunks = _split_text_safely(splitter, text)
    encoding = ingest._get_encoding()
    first_tokens = encoding.encode(chunks[0], disallowed_special=())
    second_tokens = encoding.encode(f" {chunks[1]}", disallowed_special=())
    overlaps = [
        size
        for size in range(1, min(len(first_tokens), len(second_tokens)) + 1)
        if first_tokens[-size:] == second_tokens[:size]
    ]

    assert len(chunks) > 5
    assert max(map(count_tokens, chunks)) <= CHUNK_SIZE
    assert max(overlaps) == CHUNK_OVERLAP


def test_chunk_ids_and_metadata_are_deterministic() -> None:
    documents = [Document(page_content="Texto estable.", metadata={"source": "a.md"})]

    first_chunks, first_ids = split_documents(documents, MODEL)
    second_chunks, second_ids = split_documents(documents, MODEL)

    assert first_ids == second_ids
    assert first_chunks[0].metadata == second_chunks[0].metadata
    metadata = first_chunks[0].metadata
    assert metadata["source"] == "a.md"
    assert metadata["chunk_index"] == 0
    assert metadata["embedding_model"] == MODEL
    assert metadata["chunk_id"] == deterministic_chunk_id(
        "a.md", 0, str(metadata["content_sha256"])
    )


def test_split_rejects_empty_document_sequence() -> None:
    with pytest.raises(ValueError, match="no produjo fragmentos"):
        split_documents([], MODEL)


def test_loader_rejects_file_path_and_invalid_utf8(tmp_path: Path) -> None:
    file_path = tmp_path / "not-a-directory"
    file_path.write_text("texto", encoding="utf-8")
    with pytest.raises(ValueError, match="no es un directorio"):
        load_source_documents(file_path)

    data_path = tmp_path / "data"
    data_path.mkdir()
    (data_path / "invalido.txt").write_bytes(b"\xff\xfe")
    with pytest.raises(ValueError, match="UTF-8"):
        load_source_documents(data_path)


def test_first_ingestion_persists_chunks_and_manifest(tmp_path: Path) -> None:
    data_path = tmp_path / "data"
    persist_path = tmp_path / "vectorstore"
    embeddings = DeterministicFakeEmbeddings()
    write_dataset(data_path)

    result = ingest_documents(
        data_path=data_path,
        persist_path=persist_path,
        collection_name=COLLECTION,
        embeddings=embeddings,
        embedding_model=MODEL,
    )

    recreated_ids = read_ids(persist_path, DeterministicFakeEmbeddings())
    manifest = json.loads((persist_path / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert result.status == "indexed"
    assert len(recreated_ids) == result.chunks == 1
    assert manifest["fingerprint"] == result.fingerprint
    assert embeddings.document_batches == 1


def test_unchanged_manifest_skips_embedding_work(tmp_path: Path) -> None:
    data_path = tmp_path / "data"
    persist_path = tmp_path / "vectorstore"
    write_dataset(data_path)
    ingest_documents(
        data_path=data_path,
        persist_path=persist_path,
        collection_name=COLLECTION,
        embeddings=DeterministicFakeEmbeddings(),
        embedding_model=MODEL,
    )
    second_embeddings = DeterministicFakeEmbeddings()

    result = ingest_documents(
        data_path=data_path,
        persist_path=persist_path,
        collection_name=COLLECTION,
        embeddings=second_embeddings,
        embedding_model=MODEL,
    )

    assert result.status == "skipped"
    assert second_embeddings.document_batches == 0


def test_source_change_rebuilds_without_stale_chunks(tmp_path: Path) -> None:
    data_path = tmp_path / "data"
    persist_path = tmp_path / "vectorstore"
    write_dataset(data_path, "Versión anterior que debe desaparecer.")
    ingest_documents(
        data_path=data_path,
        persist_path=persist_path,
        collection_name=COLLECTION,
        embeddings=DeterministicFakeEmbeddings(),
        embedding_model=MODEL,
    )
    old_ids = set(read_ids(persist_path, DeterministicFakeEmbeddings()))
    write_dataset(data_path, "Versión nueva y única de la fuente.")

    result = ingest_documents(
        data_path=data_path,
        persist_path=persist_path,
        collection_name=COLLECTION,
        embeddings=DeterministicFakeEmbeddings(),
        embedding_model=MODEL,
    )
    new_ids = set(read_ids(persist_path, DeterministicFakeEmbeddings()))

    assert result.status == "indexed"
    assert old_ids.isdisjoint(new_ids)
    assert len(new_ids) == result.chunks == 1


def test_replacement_embedding_failure_keeps_previous_state(tmp_path: Path) -> None:
    data_path, persist_path, previous_content, previous_manifest = seed_previous_corpus(
        tmp_path
    )
    write_dataset(data_path, "Corpus nuevo que no debe quedar activo.")

    with pytest.raises(RuntimeError, match="simulated replacement failure"):
        ingest_documents(
            data_path=data_path,
            persist_path=persist_path,
            collection_name=COLLECTION,
            embeddings=FailingDocumentEmbeddings(),
            embedding_model=MODEL,
        )

    assert_previous_state(persist_path, previous_content, previous_manifest)


def test_manifest_commit_failure_restores_previous_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_path, persist_path, previous_content, previous_manifest = seed_previous_corpus(
        tmp_path
    )
    write_dataset(data_path, "Corpus nuevo que no debe quedar activo.")

    def fail_manifest_commit(path: Path, payload: dict[str, object]) -> None:
        raise OSError("simulated manifest failure")

    monkeypatch.setattr(ingest, "_write_manifest", fail_manifest_commit)
    with pytest.raises(OSError, match="simulated manifest failure"):
        ingest_documents(
            data_path=data_path,
            persist_path=persist_path,
            collection_name=COLLECTION,
            embeddings=DeterministicFakeEmbeddings(),
            embedding_model=MODEL,
        )

    assert_previous_state(persist_path, previous_content, previous_manifest)


def test_force_rebuilds_an_unchanged_collection(tmp_path: Path) -> None:
    data_path = tmp_path / "data"
    persist_path = tmp_path / "vectorstore"
    write_dataset(data_path)
    ingest_documents(
        data_path=data_path,
        persist_path=persist_path,
        collection_name=COLLECTION,
        embeddings=DeterministicFakeEmbeddings(),
        embedding_model=MODEL,
    )
    force_embeddings = DeterministicFakeEmbeddings()

    result = ingest_documents(
        data_path=data_path,
        persist_path=persist_path,
        collection_name=COLLECTION,
        force=True,
        embeddings=force_embeddings,
        embedding_model=MODEL,
    )

    assert result.status == "indexed"
    assert force_embeddings.document_batches == 1


def test_invalid_manifest_and_invalid_persist_path_force_safe_behavior(
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "data"
    persist_path = tmp_path / "vectorstore"
    write_dataset(data_path)
    ingest_documents(
        data_path=data_path,
        persist_path=persist_path,
        collection_name=COLLECTION,
        embeddings=DeterministicFakeEmbeddings(),
        embedding_model=MODEL,
    )
    (persist_path / MANIFEST_NAME).write_text("{invalido", encoding="utf-8")

    result = ingest_documents(
        data_path=data_path,
        persist_path=persist_path,
        collection_name=COLLECTION,
        embeddings=DeterministicFakeEmbeddings(),
        embedding_model=MODEL,
    )
    assert result.status == "indexed"

    invalid_path = tmp_path / "database-file"
    invalid_path.write_text("no es un directorio", encoding="utf-8")
    with pytest.raises(ValueError, match="persistencia no es un directorio"):
        ingest_documents(
            data_path=data_path,
            persist_path=invalid_path,
            collection_name=COLLECTION,
            embeddings=DeterministicFakeEmbeddings(),
            embedding_model=MODEL,
        )


def test_default_embedding_factory_is_replaceable_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_path = tmp_path / "data"
    write_dataset(data_path)
    fake_embeddings = DeterministicFakeEmbeddings()
    selected_models: list[str] = []

    def fake_factory(*, model: str) -> DeterministicFakeEmbeddings:
        selected_models.append(model)
        return fake_embeddings

    monkeypatch.setenv("OPENAI_API_KEY", "sk-placeholder-for-test")
    monkeypatch.setattr(ingest, "OpenAIEmbeddings", fake_factory)

    result = ingest_documents(
        data_path=data_path,
        persist_path=tmp_path / "vectorstore",
        collection_name=COLLECTION,
        embedding_model=MODEL,
    )

    assert result.status == "indexed"
    assert selected_models == [MODEL]
