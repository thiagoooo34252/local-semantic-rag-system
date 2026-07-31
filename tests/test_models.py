from __future__ import annotations

import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import ValidationError

from models import FALLBACK_ANSWER, RagResponse


def test_rejects_blank_answer() -> None:
    with pytest.raises(ValidationError, match="no puede estar vacía"):
        RagResponse(answer="  ", references=[])


def test_rejects_fallback_with_references() -> None:
    parser = PydanticOutputParser(pydantic_object=RagResponse)

    with pytest.raises(OutputParserException, match="referencias vacías"):
        parser.parse('{"answer": "No lo sé", "references": ["fuente-inventada.md"]}')


def test_normalizes_unique_references_in_order() -> None:
    response = RagResponse.model_validate(
        {
            "answer": "Respuesta sustentada.",
            "references": [" a.md ", "a.md", "", "b.txt", "a.md"],
        }
    )

    assert response.references == ["a.md", "b.txt"]


def test_accepts_exact_fallback_without_references() -> None:
    response = RagResponse(answer=FALLBACK_ANSWER, references=[])

    assert response.model_dump() == {"answer": "No lo sé", "references": []}
