"""Внутреннее представление Forge (IR).

Всё, что Forge узнаёт о будущем агенте, накапливается в одном объекте
:class:`~forge.model.blueprint.AgentBlueprint`. Это источник истины, который
читают и изменяют этапы приёма, уточнения, фиксации контракта, синтеза,
тестирования и упаковки, и именно он сохраняется между вызовами CLI.
"""

from __future__ import annotations

from .blueprint import (
    AgentBlueprint,
    Assumption,
    Capability,
    FieldSpec,
    InputSpec,
    IOContract,
    HistoryEvent,
    Issue,
    IssueKind,
    IssueStatus,
    OutputKind,
    Question,
    Severity,
    Stage,
    StructuredInstructions,
    TestCase,
)

__all__ = [
    "AgentBlueprint",
    "Assumption",
    "Capability",
    "FieldSpec",
    "HistoryEvent",
    "InputSpec",
    "IOContract",
    "Issue",
    "IssueKind",
    "IssueStatus",
    "OutputKind",
    "Question",
    "Severity",
    "Stage",
    "StructuredInstructions",
    "TestCase",
]
