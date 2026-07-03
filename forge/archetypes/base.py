"""Контракт архетипа + реестр.

*Архетип* — это вид создаваемого агента (извлечение, классификация,
проверка рисков, сравнение пунктов, генерация документов, чат, рабочий процесс).
Это главная точка расширения: каждый архетип декларативен — значения по
умолчанию, базовые уточняющие вопросы и имя *стратегии исполнения*, которую
будет использовать сгенерированный агент.

Чтобы добавить новый архетип: создайте подкласс :class:`AgentArchetype`,
задайте его поля, примените ``@register``. В конвейере ничего не меняется.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..model import (
    AgentBlueprint,
    Capability,
    FieldSpec,
    OutputKind,
    Question,
)


@dataclass
class AgentArchetype:
    key: str
    title: str
    description: str
    runtime_strategy: str                       # какой исполнитель времени выполнения генерировать
    output_kind: OutputKind = OutputKind.JSON
    capabilities: tuple[Capability, ...] = ()
    default_fields: tuple[FieldSpec, ...] = ()
    # Число документов на входе, которое ожидает стратегия (0 = только текст/json).
    input_arity: int = 1
    # Нужна ли в выходе самооценка уверенности (детерминированные/пакетные — нет).
    confidence: bool = True
    # (текст вопроса, зачем, category, blocking)
    clarifiers: tuple[tuple[str, str, str, bool], ...] = ()

    def baseline_questions(self, bp: AgentBlueprint) -> list[Question]:
        out: list[Question] = []
        for i, (text, why, cat, blocking) in enumerate(self.clarifiers, start=1):
            out.append(
                Question(
                    id=f"Q-ARCH-{self.key}-{i:02d}",
                    text=text,
                    why=why,
                    category=cat,
                    blocking=blocking,
                )
            )
        return out

    def seed_contract(self, bp: AgentBlueprint) -> None:
        """Применить значения архетипа по умолчанию к чертежу, у которого их ещё нет."""
        if not bp.capabilities:
            bp.capabilities = list(self.capabilities)
        bp.io.output_kind = self.output_kind
        bp.io.confidence_required = self.confidence
        if not bp.io.fields and self.default_fields:
            bp.io.fields = [f.model_copy(deep=True) for f in self.default_fields]


# --------------------------------------------------------------------------- #
# Реестр
# --------------------------------------------------------------------------- #
_REGISTRY: dict[str, AgentArchetype] = {}


def register(arch: AgentArchetype) -> AgentArchetype:
    _REGISTRY[arch.key] = arch
    return arch


def get(key: str) -> AgentArchetype:
    if key not in _REGISTRY:
        raise KeyError(f"Неизвестный архетип '{key}'. Известные: {', '.join(sorted(_REGISTRY))}")
    return _REGISTRY[key]


def all_archetypes() -> list[AgentArchetype]:
    return list(_REGISTRY.values())


def keys() -> list[str]:
    return sorted(_REGISTRY)
