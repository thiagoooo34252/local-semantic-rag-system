from __future__ import annotations

from pathlib import Path

import pytest

import settings


def test_environment_getters_use_configured_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "embedding-configurado")
    monkeypatch.setenv("OPENAI_CHAT_MODEL", "chat-configurado")
    monkeypatch.setenv("RAG_PERSIST_PATH", "otra-base")
    monkeypatch.setenv("RAG_COLLECTION_NAME", "otra-coleccion")
    monkeypatch.setenv("RAG_TOP_K", "5")

    assert settings.get_embedding_model() == "embedding-configurado"
    assert settings.get_chat_model() == "chat-configurado"
    assert settings.get_persist_path() == Path("otra-base")
    assert settings.get_collection_name() == "otra-coleccion"
    assert settings.get_top_k() == 5


@pytest.mark.parametrize("configured_value", [None, "", "  \t"])
def test_reasoning_effort_is_disabled_when_unset_or_blank(
    monkeypatch: pytest.MonkeyPatch,
    configured_value: str | None,
) -> None:
    if configured_value is None:
        monkeypatch.delenv("OPENAI_REASONING_EFFORT", raising=False)
    else:
        monkeypatch.setenv("OPENAI_REASONING_EFFORT", configured_value)

    assert settings.get_reasoning_effort() is None


def test_invalid_reasoning_effort_has_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", "minimal")

    with pytest.raises(
        ValueError,
        match=(
            "OPENAI_REASONING_EFFORT debe ser uno de: "
            "none, low, medium, high, xhigh, max"
        ),
    ):
        settings.get_reasoning_effort()


def test_invalid_top_k_configuration_has_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG_TOP_K", "cuatro")

    with pytest.raises(ValueError, match="número entero"):
        settings.get_top_k()


def test_dotenv_can_be_enabled_or_explicitly_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bool] = []
    monkeypatch.setattr(settings, "load_dotenv", lambda: calls.append(True))
    monkeypatch.delenv("PYTHON_DOTENV_DISABLED", raising=False)
    settings.load_project_environment()
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    settings.load_project_environment()

    assert calls == [True]


def test_api_key_validation_never_includes_a_key_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError) as captured:
        settings.require_openai_api_key()
    assert "OPENAI_API_KEY" in str(captured.value)

    monkeypatch.setenv("OPENAI_API_KEY", "sk-placeholder-for-test")
    settings.require_openai_api_key()
