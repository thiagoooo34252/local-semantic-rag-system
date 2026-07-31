"""Modelos de salida validados para el sistema RAG."""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

FALLBACK_ANSWER = "No lo sé"


class RagResponse(BaseModel):
    """Respuesta fundamentada y fuentes del contexto recuperado."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(
        description="Respuesta sustentada exclusivamente por el contexto."
    )
    references: list[str] = Field(
        default_factory=list,
        description="Nombres de archivo que sustentan la respuesta.",
    )

    @field_validator("answer")
    @classmethod
    def validate_answer(cls, value: str) -> str:
        """Normaliza la respuesta y rechaza texto vacío."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("La respuesta no puede estar vacía.")
        return normalized

    @field_validator("references")
    @classmethod
    def normalize_references(cls, value: list[str]) -> list[str]:
        """Normaliza y elimina referencias repetidas preservando el orden."""
        normalized: list[str] = []
        seen: set[str] = set()
        for reference in value:
            candidate = reference.strip()
            if candidate and candidate not in seen:
                normalized.append(candidate)
                seen.add(candidate)
        return normalized

    @model_validator(mode="after")
    def validate_fallback(self) -> Self:
        """Exige que la respuesta de desconocimiento no incluya referencias."""
        if self.answer == FALLBACK_ANSWER and self.references:
            raise ValueError(f"'{FALLBACK_ANSWER}' debe tener referencias vacías.")
        return self
