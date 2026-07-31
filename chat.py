"""Interfaz conversacional segura para el sistema RAG local."""

from __future__ import annotations

import asyncio
import os
import sys
import unicodedata
from collections.abc import Awaitable, Callable, Mapping
from enum import Enum, auto
from typing import TextIO

from models import RagResponse
from rag import (
    ProgressCallback,
    RagPhase,
    RagProgressEvent,
    get_rag_response_with_progress,
)
from settings import (
    get_chat_model,
    get_reasoning_effort,
    load_project_environment,
)

PROMPT = "\N{SINGLE RIGHT-POINTING ANGLE QUOTATION MARK} "

InputCallable = Callable[[str], str]
ResponseCallable = Callable[[str, ProgressCallback], Awaitable[RagResponse]]

PHASE_LABELS = {
    RagPhase.LOCAL_RETRIEVAL: "Local retrieval",
    RagPhase.GROUNDED_GENERATION_AND_PARSING: ("Grounded generation and parsing"),
    RagPhase.REFERENCE_VALIDATION: "Reference validation",
}


class Command(Enum):
    """Comandos reconocidos por la sesión interactiva."""

    HELP = auto()
    CLEAR = auto()
    EXIT = auto()
    UNKNOWN = auto()


COMMANDS = {
    "/help": Command.HELP,
    "/ayuda": Command.HELP,
    "/?": Command.HELP,
    "/clear": Command.CLEAR,
    "/limpiar": Command.CLEAR,
    "/exit": Command.EXIT,
    "/salir": Command.EXIT,
    "/quit": Command.EXIT,
    "/q": Command.EXIT,
}


def sanitize_terminal_text(value: str) -> str:
    """Elimina controles capaces de alterar la terminal."""
    normalized = value.replace("\r\n", "\n").replace("\r", "")
    return "".join(
        character
        for character in normalized
        if character in {"\n", "\t"} or unicodedata.category(character) != "Cc"
    )


def _styled(value: str, code: str, use_ansi: bool) -> str:
    return f"\033[{code}m{value}\033[0m" if use_ansi else value


def render_welcome(model: str, reasoning_effort: str | None, *, use_ansi: bool) -> str:
    """Construye el encabezado compacto de la conversación."""
    safe_model = sanitize_terminal_text(model)
    configuration = f"Modelo: {safe_model}"
    if reasoning_effort is not None:
        safe_effort = sanitize_terminal_text(reasoning_effort)
        configuration += f" · Razonamiento: {safe_effort}"
    return "\n".join(
        [
            _styled("Local RAG", "1;36", use_ansi),
            _styled(configuration, "2", use_ansi),
            "Pregunta sobre los documentos indexados. /help muestra los comandos.",
        ]
    )


def render_help(*, use_ansi: bool) -> str:
    """Muestra únicamente los controles disponibles para la sesión."""
    return "\n".join(
        [
            _styled("Comandos", "1", use_ansi),
            "/help, /ayuda     Mostrar esta ayuda",
            "/clear, /limpiar  Limpiar la conversación",
            "/exit, /salir     Salir",
        ]
    )


def render_response(response: RagResponse, *, use_ansi: bool) -> str:
    """Presenta una respuesta validada como texto natural y fuentes separadas."""
    answer = sanitize_terminal_text(response.answer)
    if not response.references:
        return answer
    references = " · ".join(
        sanitize_terminal_text(reference) for reference in response.references
    )
    source_label = _styled("Fuentes:", "2", use_ansi)
    return f"{answer}\n\n{source_label} {references}"


def format_duration(elapsed_seconds: float) -> str:
    """Formatea una duración monotónica con precisión compacta."""
    safe_elapsed = max(0.0, elapsed_seconds)
    if safe_elapsed < 0.9995:
        milliseconds = int(safe_elapsed * 1000 + 0.5)
        return f"{milliseconds}ms"
    seconds = round(safe_elapsed, 1)
    return f"{seconds:g}s"


def render_activity(event: RagProgressEvent, *, use_ansi: bool) -> str:
    """Presenta una fase operacional completada sin contenido de razonamiento."""
    label = PHASE_LABELS[event.phase]
    line = f"Activity: {label} · {format_duration(event.elapsed_seconds)}"
    return _styled(line, "38;5;214", use_ansi)


def render_error(error: Exception, *, use_ansi: bool) -> str:
    """Traduce categorías de error sin interpolar detalles potencialmente sensibles."""
    if isinstance(error, TimeoutError):
        message = "La consulta tardó demasiado. Inténtalo de nuevo."
    elif isinstance(error, ValueError):
        message = "No pude procesar la consulta. Revisa la configuración."
    elif isinstance(error, RuntimeError):
        message = (
            "No pude completar la consulta. Verifica la configuración y la base "
            "vectorial."
        )
    else:
        message = "No pude completar la consulta. Inténtalo de nuevo."
    return f"{_styled('Error:', '31', use_ansi)} {message}"


def parse_command(value: str) -> Command | None:
    """Reconoce comandos completos sin confundir preguntas normales."""
    normalized = value.strip().casefold()
    if not normalized.startswith("/"):
        return None
    return COMMANDS.get(normalized, Command.UNKNOWN)


def supports_ansi(stream: TextIO, environ: Mapping[str, str]) -> bool:
    """Habilita ANSI solo para terminales que no solicitan `NO_COLOR`."""
    return "NO_COLOR" not in environ and stream.isatty()


def render_clear(*, use_ansi: bool) -> str:
    """Limpia una TTY y produce una separación inocua en salida redirigida."""
    return "\033[2J\033[H" if use_ansi else "\n"


def _write(output: TextIO, value: str) -> None:
    output.write(value)
    output.flush()


async def _request_with_activity(
    query: str,
    response_callable: ResponseCallable,
    output: TextIO,
    *,
    use_ansi: bool,
) -> RagResponse:
    def report_activity(event: RagProgressEvent) -> None:
        _write(output, f"{render_activity(event, use_ansi=use_ansi)}\n")

    return await response_callable(query, report_activity)


async def run_chat(
    response_callable: ResponseCallable,
    *,
    model: str,
    reasoning_effort: str | None,
    input_callable: InputCallable = input,
    output: TextIO | None = None,
    environ: Mapping[str, str] | None = None,
) -> None:
    """Ejecuta una sesión conversacional con dependencias inyectables."""
    target = output if output is not None else sys.stdout
    environment = environ if environ is not None else os.environ
    use_ansi = supports_ansi(target, environment)
    welcome = render_welcome(model, reasoning_effort, use_ansi=use_ansi)
    _write(target, f"{welcome}\n\n")

    while True:
        try:
            value = input_callable(PROMPT)
        except (EOFError, KeyboardInterrupt):
            _write(target, "\nHasta luego.\n")
            return

        query = value.strip()
        if not query:
            continue

        command = parse_command(query)
        if command is Command.EXIT:
            _write(target, "Hasta luego.\n")
            return
        if command is Command.HELP:
            _write(target, f"\n{render_help(use_ansi=use_ansi)}\n\n")
            continue
        if command is Command.CLEAR:
            _write(target, f"{render_clear(use_ansi=use_ansi)}{welcome}\n\n")
            continue
        if command is Command.UNKNOWN:
            _write(target, "Comando desconocido. Usa /help para ver las opciones.\n\n")
            continue

        try:
            response = await _request_with_activity(
                query, response_callable, target, use_ansi=use_ansi
            )
        except Exception as error:
            _write(target, f"{render_error(error, use_ansi=use_ansi)}\n\n")
            continue
        _write(target, f"\n{render_response(response, use_ansi=use_ansi)}\n\n")


def main() -> None:
    """Carga la configuración e inicia el chat sin exponer errores internos."""
    try:
        load_project_environment()
        model = get_chat_model()
        reasoning_effort = get_reasoning_effort()
        asyncio.run(
            run_chat(
                get_rag_response_with_progress,
                model=model,
                reasoning_effort=reasoning_effort,
            )
        )
    except KeyboardInterrupt:
        print("\nHasta luego.")
    except Exception as error:
        use_ansi = supports_ansi(sys.stderr, os.environ)
        print(render_error(error, use_ansi=use_ansi), file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
