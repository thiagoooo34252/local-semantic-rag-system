from __future__ import annotations

import io
from collections.abc import Iterator

import pytest

import chat
from chat import Command
from models import FALLBACK_ANSWER, RagResponse
from rag import ProgressCallback, RagPhase, RagProgressEvent


class TtyBuffer(io.StringIO):
    def isatty(self) -> bool:
        return True


def input_from(
    values: list[str | BaseException],
) -> tuple[chat.InputCallable, list[str]]:
    remaining: Iterator[str | BaseException] = iter(values)
    prompts: list[str] = []

    def read_input(prompt: str) -> str:
        prompts.append(prompt)
        value = next(remaining)
        if isinstance(value, BaseException):
            raise value
        return value

    return read_input, prompts


def emit_completed_phases(callback: ProgressCallback) -> None:
    callback(RagProgressEvent(RagPhase.LOCAL_RETRIEVAL, 0.057))
    callback(RagProgressEvent(RagPhase.GROUNDED_GENERATION_AND_PARSING, 2.1))
    callback(RagProgressEvent(RagPhase.REFERENCE_VALIDATION, 0.002))


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("/help", Command.HELP),
        (" /AYUDA ", Command.HELP),
        ("/clear", Command.CLEAR),
        ("/limpiar", Command.CLEAR),
        ("/exit", Command.EXIT),
        ("/salir", Command.EXIT),
        ("/desconocido", Command.UNKNOWN),
        ("¿Qué sendero es accesible?", None),
    ],
)
def test_parse_command_recognizes_commands_and_spanish_aliases(
    value: str, expected: Command | None
) -> None:
    assert chat.parse_command(value) is expected


def test_welcome_shows_model_and_configured_reasoning_effort() -> None:
    rendered = chat.render_welcome("gpt-5.6-luna", "low", use_ansi=False)

    assert "Local RAG" in rendered
    assert "gpt-5.6-luna" in rendered
    assert "Razonamiento: low" in rendered
    assert "\x1b" not in rendered


def test_welcome_omits_reasoning_when_not_configured() -> None:
    rendered = chat.render_welcome("gpt-4o-mini", None, use_ansi=False)

    assert "gpt-4o-mini" in rendered
    assert "Razonamiento" not in rendered


def test_response_renders_answer_and_compact_sources_without_json() -> None:
    rendered = chat.render_response(
        RagResponse(
            answer="La pasarela dura 45 minutos.",
            references=["senderos.md", "planificacion.txt"],
        ),
        use_ansi=False,
    )

    assert rendered == (
        "La pasarela dura 45 minutos.\n\nFuentes: senderos.md · planificacion.txt"
    )
    assert '"answer"' not in rendered
    assert "{" not in rendered


def test_fallback_renders_without_an_empty_sources_label() -> None:
    rendered = chat.render_response(
        RagResponse(answer=FALLBACK_ANSWER, references=[]), use_ansi=False
    )

    assert rendered == FALLBACK_ANSWER
    assert "Fuentes" not in rendered


def test_rendering_removes_untrusted_terminal_control_characters() -> None:
    rendered = chat.render_response(
        RagResponse(
            answer="Respuesta\x1b[2J segura\r.",
            references=["fuente\x1b]8;;https://example.com\x07.md"],
        ),
        use_ansi=False,
    )

    assert "\x1b" not in rendered
    assert "\x07" not in rendered
    assert "\r" not in rendered
    assert "Respuesta[2J segura." in rendered


def test_ansi_requires_tty_and_is_disabled_by_no_color() -> None:
    assert chat.supports_ansi(TtyBuffer(), {}) is True
    assert chat.supports_ansi(TtyBuffer(), {"NO_COLOR": ""}) is False
    assert chat.supports_ansi(io.StringIO(), {}) is False


def test_non_tty_rendering_and_clear_never_emit_ansi() -> None:
    assert "\x1b" not in chat.render_welcome("gpt-test", None, use_ansi=False)
    assert "\x1b" not in chat.render_help(use_ansi=False)
    assert "\x1b" not in chat.render_clear(use_ansi=False)
    assert "\x1b" not in chat.render_activity(
        RagProgressEvent(RagPhase.LOCAL_RETRIEVAL, 0.057), use_ansi=False
    )


@pytest.mark.parametrize(
    ("elapsed_seconds", "expected"),
    [
        (-1.0, "0ms"),
        (0.0, "0ms"),
        (0.0566, "57ms"),
        (0.9994, "999ms"),
        (0.9995, "1s"),
        (1.0, "1s"),
        (2.06, "2.1s"),
    ],
)
def test_duration_formatting_boundaries(elapsed_seconds: float, expected: str) -> None:
    assert chat.format_duration(elapsed_seconds) == expected


def test_activity_uses_amber_on_tty_and_plain_text_without_ansi() -> None:
    event = RagProgressEvent(RagPhase.LOCAL_RETRIEVAL, 0.057)

    assert chat.render_activity(event, use_ansi=False) == (
        "Activity: Local retrieval · 57ms"
    )
    assert chat.render_activity(event, use_ansi=True) == (
        "\x1b[38;5;214mActivity: Local retrieval · 57ms\x1b[0m"
    )


def test_activity_contract_contains_no_private_reasoning_fields_or_language() -> None:
    assert set(RagProgressEvent.__dataclass_fields__) == {
        "phase",
        "elapsed_seconds",
    }
    rendered = "\n".join(
        chat.render_activity(RagProgressEvent(phase, 0.1), use_ansi=False)
        for phase in RagPhase
    ).casefold()

    for forbidden in (
        "thought",
        "reasoning",
        "chain-of-thought",
        "plan",
        "summary",
        "private",
    ):
        assert forbidden not in rendered


@pytest.mark.asyncio
async def test_chat_calls_async_rag_and_renders_natural_response() -> None:
    queries: list[str] = []
    output = io.StringIO()
    read_input, prompts = input_from(["  ¿Qué sendero es accesible?  ", "/exit"])

    async def respond(query: str, on_progress: ProgressCallback) -> RagResponse:
        queries.append(query)
        emit_completed_phases(on_progress)
        return RagResponse(
            answer="La Pasarela del Humedal es accesible.",
            references=["02_senderos_y_accesibilidad.txt"],
        )

    await chat.run_chat(
        respond,
        model="gpt-test",
        reasoning_effort=None,
        input_callable=read_input,
        output=output,
        environ={},
    )

    rendered = output.getvalue()
    assert queries == ["¿Qué sendero es accesible?"]
    assert prompts == [
        "\N{SINGLE RIGHT-POINTING ANGLE QUOTATION MARK} ",
        "\N{SINGLE RIGHT-POINTING ANGLE QUOTATION MARK} ",
    ]
    assert [line for line in rendered.splitlines() if line.startswith("Activity:")] == [
        "Activity: Local retrieval · 57ms",
        "Activity: Grounded generation and parsing · 2.1s",
        "Activity: Reference validation · 2ms",
    ]
    assert rendered.index("Activity: Reference validation") < rendered.index(
        "La Pasarela del Humedal es accesible."
    )
    assert "La Pasarela del Humedal es accesible." in rendered
    assert "Fuentes: 02_senderos_y_accesibilidad.txt" in rendered
    assert '"answer"' not in rendered
    assert "Hasta luego." in rendered


@pytest.mark.asyncio
async def test_commands_do_not_call_rag_and_clear_degrades_without_ansi() -> None:
    calls = 0
    output = io.StringIO()
    read_input, _prompts = input_from(
        ["/ayuda", "/limpiar", "/desconocido", "   ", "/salir"]
    )

    async def respond(_query: str, _on_progress: ProgressCallback) -> RagResponse:
        nonlocal calls
        calls += 1
        return RagResponse(answer=FALLBACK_ANSWER, references=[])

    await chat.run_chat(
        respond,
        model="gpt-test",
        reasoning_effort="none",
        input_callable=read_input,
        output=output,
        environ={"NO_COLOR": "1"},
    )

    rendered = output.getvalue()
    assert calls == 0
    assert "/help, /ayuda" in rendered
    assert rendered.count("Local RAG") == 2
    assert "Comando desconocido" in rendered
    assert "\x1b" not in rendered


@pytest.mark.asyncio
@pytest.mark.parametrize("interruption", [EOFError(), KeyboardInterrupt()])
async def test_input_interruptions_exit_without_traceback(
    interruption: BaseException,
) -> None:
    output = io.StringIO()
    read_input, _prompts = input_from([interruption])

    async def respond(_query: str, _on_progress: ProgressCallback) -> RagResponse:
        raise AssertionError("No debe consultar el RAG.")

    await chat.run_chat(
        respond,
        model="gpt-test",
        reasoning_effort=None,
        input_callable=read_input,
        output=output,
        environ={},
    )

    rendered = output.getvalue()
    assert "Hasta luego." in rendered
    assert "Traceback" not in rendered


@pytest.mark.asyncio
async def test_query_errors_are_sanitized_and_session_remains_usable() -> None:
    secret = "sk-sensitive-value-that-must-not-appear"
    output = io.StringIO()
    read_input, _prompts = input_from(["pregunta", "/exit"])

    async def fail(_query: str, _on_progress: ProgressCallback) -> RagResponse:
        raise RuntimeError(f"provider rejected credential {secret}")

    await chat.run_chat(
        fail,
        model="gpt-test",
        reasoning_effort=None,
        input_callable=read_input,
        output=output,
        environ={},
    )

    rendered = output.getvalue()
    assert "No pude completar la consulta" in rendered
    assert secret not in rendered
    assert "provider rejected" not in rendered
    assert "Traceback" not in rendered
    assert "Hasta luego." in rendered


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (TimeoutError("secret"), "tardó demasiado"),
        (ValueError("secret"), "Revisa la configuración"),
        (RuntimeError("secret"), "base vectorial"),
        (OSError("secret"), "No pude completar la consulta"),
    ],
)
def test_error_messages_are_actionable_without_using_exception_text(
    error: Exception, expected: str
) -> None:
    rendered = chat.render_error(error, use_ansi=False)

    assert expected in rendered
    assert "secret" not in rendered


@pytest.mark.asyncio
async def test_tty_status_and_clear_use_ansi() -> None:
    output = TtyBuffer()
    read_input, _prompts = input_from(["pregunta", "/clear", "/exit"])

    async def respond(_query: str, on_progress: ProgressCallback) -> RagResponse:
        emit_completed_phases(on_progress)
        return RagResponse(answer=FALLBACK_ANSWER, references=[])

    await chat.run_chat(
        respond,
        model="gpt-test",
        reasoning_effort=None,
        input_callable=read_input,
        output=output,
        environ={},
    )

    rendered = output.getvalue()
    assert "\x1b[38;5;214mActivity: Local retrieval · 57ms\x1b[0m" in rendered
    assert "\x1b[2J\x1b[H" in rendered


@pytest.mark.asyncio
async def test_no_color_tty_renders_accumulated_activity_as_plain_text() -> None:
    output = TtyBuffer()
    read_input, _prompts = input_from(["pregunta", "/exit"])

    async def respond(_query: str, on_progress: ProgressCallback) -> RagResponse:
        emit_completed_phases(on_progress)
        return RagResponse(answer=FALLBACK_ANSWER, references=[])

    await chat.run_chat(
        respond,
        model="gpt-test",
        reasoning_effort=None,
        input_callable=read_input,
        output=output,
        environ={"NO_COLOR": ""},
    )

    rendered = output.getvalue()
    assert "Activity: Local retrieval · 57ms" in rendered
    assert "Activity: Grounded generation and parsing · 2.1s" in rendered
    assert "Activity: Reference validation · 2ms" in rendered
    assert "\x1b" not in rendered


def test_main_sanitizes_startup_errors(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    secret = "sk-startup-secret"

    def fail_startup() -> None:
        raise RuntimeError(f"invalid credential {secret}")

    monkeypatch.setattr(chat, "load_project_environment", fail_startup)

    with pytest.raises(SystemExit) as captured:
        chat.main()

    output = capsys.readouterr()
    assert captured.value.code == 1
    assert "No pude completar la consulta" in output.err
    assert secret not in output.err
    assert "Traceback" not in output.err
