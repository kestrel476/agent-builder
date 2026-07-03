"""Критики: детерминированные + LLM-проверки, превращающие слабые места чертежа
в явные объекты :class:`Issue`.

Руководящее правило из задания: *обнаруживай неоднозначность, никогда не угадывай
молча.* Поэтому критики намеренно подозрительны. Каждое замечание несёт
``rationale`` (почему это важно) и, где возможно, ``suggestion`` (безопасное
значение по умолчанию, которое пользователь может принять) — оно позже становится
зафиксированным допущением, если остаётся без ответа.
"""

from __future__ import annotations

import re
from typing import Iterable

from ..integrations import ForgeLLM
from ..model import AgentBlueprint, Issue, IssueKind, Severity

# Размытые формулировки (англ. и рус.), указывающие на неоднозначность.
_VAGUE = re.compile(
    r"\b(reasonable|as appropriate|where applicable|relevant|etc\.?|and so on|various|"
    r"some|if needed|appropriately|properly|adequate|и т\.?д|при необходимости|разумн)\b",
    re.I,
)


class Critic:
    """Базовый критик. ``run`` возвращает замечания (ещё без устранения дубликатов)."""

    name = "critic"

    def run(self, bp: AgentBlueprint) -> list[Issue]:  # pragma: no cover - интерфейс
        raise NotImplementedError


def _issue(bp: AgentBlueprint, kind: IssueKind, sev: Severity, message: str, *,
           where: str = "", rationale: str = "", suggestion: str = "", source: str = "") -> Issue:
    return Issue(id=bp.next_id("ISS"), kind=kind, severity=sev, message=message,
                 where=where, rationale=rationale, suggestion=suggestion, source=source)


# --------------------------------------------------------------------------- #
# Детерминированные критики
# --------------------------------------------------------------------------- #
class GoalCritic(Critic):
    name = "goal"

    def run(self, bp):
        if not (bp.goal or bp.instructions.goal).strip():
            return [_issue(bp, IssueKind.MISSING, Severity.BLOCKER,
                           "У агента не указана цель.", where="goal",
                           rationale="Без цели в одно предложение область применения не определена.",
                           suggestion="Сформулируйте единственную задачу, которую должен выполнять этот агент.", source=self.name)]
        return []


class AmbiguityCritic(Critic):
    name = "ambiguity"

    def run(self, bp):
        issues: list[Issue] = []
        seen: set[str] = set()
        sources: Iterable[str] = (
            [bp.goal, bp.domain_notes]
            + bp.instructions.business_rules
            + bp.instructions.constraints
            + bp.instructions.ambiguities
        )
        for text in sources:
            for m in _VAGUE.finditer(text or ""):
                phrase = m.group(0).lower()
                if phrase in seen:
                    continue
                seen.add(phrase)
                issues.append(_issue(
                    bp, IssueKind.AMBIGUITY, Severity.MAJOR,
                    f'Размытая формулировка "{m.group(0)}" в: "{_clip(text)}".', where="instructions",
                    rationale="Неконкретная формулировка приводит к невоспроизводимому поведению агента.",
                    suggestion="Замените точным, проверяемым критерием.", source=self.name))
        return issues


class MissingDataCritic(Critic):
    name = "missing"

    def run(self, bp):
        return [
            _issue(bp, IssueKind.MISSING, Severity.MAJOR, miss, where="instructions.missing_data",
                   rationale="Упоминается, но нигде не указано.", source=self.name)
            for miss in bp.instructions.missing_data
        ]


class ContractCritic(Critic):
    name = "contract"

    def run(self, bp):
        issues: list[Issue] = []
        io = bp.io
        if io.output_kind.value in ("json", "classification", "report") and not io.fields:
            issues.append(_issue(bp, IssueKind.MISSING, Severity.BLOCKER,
                                 "У схемы выхода нет полей.", where="io.fields",
                                 rationale="Структурированный агент без схемы выхода нельзя построить или протестировать.",
                                 suggestion="Определите каждое выходное поле, его тип и является ли оно обязательным.",
                                 source=self.name))
        if io.fields and not any(f.required for f in io.fields):
            issues.append(_issue(bp, IssueKind.MISSING, Severity.MAJOR,
                                 "Ни одно выходное поле не отмечено как обязательное.", where="io.fields",
                                 rationale="Если ничего не обязательно, пустой результат тривиально «проходит».",
                                 suggestion="Отметьте поля, которые должны присутствовать всегда.", source=self.name))
        if not io.inputs:
            issues.append(_issue(bp, IssueKind.MISSING, Severity.BLOCKER,
                                 "Не определён ни один вход.", where="io.inputs",
                                 rationale="Входная поверхность агента не указана.",
                                 suggestion="Укажите, что получает агент (текст, JSON, файл документа).",
                                 source=self.name))
        if not io.error_definition.strip():
            issues.append(_issue(bp, IssueKind.MISSING, Severity.MAJOR,
                                 "Нет определения того, что считается ошибкой.", where="io.error_definition",
                                 rationale="Ошибка, частичный результат и успех должны быть различимы.",
                                 suggestion="Определите исходы ошибки, частичного результата и низкой уверенности.",
                                 source=self.name))
        return issues


class VagueSuccessCritic(Critic):
    name = "success"

    def run(self, bp):
        crit = bp.io.success_criteria
        if not crit:
            return [_issue(bp, IssueKind.VAGUE_SUCCESS, Severity.BLOCKER,
                           "Критерии успеха не определены.", where="io.success_criteria",
                           rationale="Без измеримых критериев успеха агента нельзя проверить.",
                           suggestion="Укажите для каждого поля или в целом, что делает вывод правильным.",
                           source=self.name)]
        issues = []
        for c in crit:
            if _VAGUE.search(c):
                issues.append(_issue(bp, IssueKind.VAGUE_SUCCESS, Severity.MAJOR,
                                     f'Критерий успеха размыт: "{_clip(c)}".', where="io.success_criteria",
                                     rationale="Нечёткий критерий нельзя превратить в проходящий/непроходящий тест.",
                                     suggestion="Сделайте его конкретным и проверяемым.", source=self.name))
        return issues


class ExampleConflictCritic(Critic):
    """Перекрёстная проверка тест-кейсов против контракта — классический конфликт примера и спецификации."""

    name = "examples"

    def run(self, bp):
        issues: list[Issue] = []
        field_names = {f.name for f in bp.io.fields}
        required = {f.name for f in bp.io.fields if f.required}
        for tc in bp.test_cases:
            if not tc.expected:
                continue
            extra = set(tc.expected) - field_names - {"confidence"}
            if extra and field_names:
                issues.append(_issue(
                    bp, IssueKind.EXAMPLE_CONFLICT, Severity.MAJOR,
                    f"Тест '{tc.name}' ожидает поле(я) {sorted(extra)}, которых нет в схеме выхода.",
                    where=f"test_cases.{tc.id}",
                    rationale="Пример противоречит контракту; что-то из них неверно.",
                    suggestion="Добавьте поле в схему или исправьте ожидаемый вывод.", source=self.name))
            missing_req = required - set(tc.expected)
            if missing_req:
                issues.append(_issue(
                    bp, IssueKind.EXAMPLE_CONFLICT, Severity.MINOR,
                    f"Тест '{tc.name}' опускает обязательное(ые) поле(я) {sorted(missing_req)} в своём ожидаемом выводе.",
                    where=f"test_cases.{tc.id}",
                    rationale="Либо поле на самом деле не обязательное, либо пример неполон.",
                    suggestion="Сделайте ожидаемый вывод полным или снимите флаг 'required' у поля.",
                    source=self.name))
        return issues


class CapabilityCritic(Critic):
    """Выявляет пробелы в возможностях, подразумеваемые контрактом."""

    name = "capabilities"

    def run(self, bp):
        from ..model import Capability
        issues = []
        wants_docs = any(i.kind == "document" for i in bp.io.inputs)
        if wants_docs and Capability.DOC_EXTRACTION not in bp.capabilities:
            issues.append(_issue(bp, IssueKind.RISK, Severity.MAJOR,
                                 "Объявлен ввод документа, но возможность извлечения из документов отключена.",
                                 where="capabilities",
                                 rationale="Агент не может прочитать свои входные данные.",
                                 suggestion="Включите doc_extraction (и OCR для сканов).", source=self.name))
        return issues


DETERMINISTIC_CRITICS: list[Critic] = [
    GoalCritic(), AmbiguityCritic(), MissingDataCritic(), ContractCritic(),
    VagueSuccessCritic(), ExampleConflictCritic(), CapabilityCritic(),
]


# --------------------------------------------------------------------------- #
# LLM-критик — предлагает более глубокие вопросы, которых не видят эвристики
# --------------------------------------------------------------------------- #
_QUESTIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "why": {"type": "string"},
                    "category": {"type": "string"},
                    "blocking": {"type": "boolean"},
                },
                "required": ["text", "why"],
            },
        }
    },
    "required": ["questions"],
}


def llm_questions(bp: AgentBlueprint, llm: ForgeLLM, *, limit: int = 5) -> list[dict]:
    """Запросить у «мозга» дополнительные ценные уточняющие вопросы."""
    system = (
        "Вы — старший инженер по требованиям для агентов автоматизации юридических задач. "
        "По черновой спецификации выявите наиболее ценные уточняющие вопросы, "
        "которые устраняют неоднозначность до того, как будет написан код. Предпочитайте вопросы, "
        "ответы на которые меняют схему выхода, критерии успеха или политику обработки ошибок. "
        "Не спрашивайте о том, что уже указано."
    )
    user = (
        f"Цель: {bp.goal}\nАрхетип: {bp.archetype}\n"
        f"Бизнес-правила: {bp.instructions.business_rules}\n"
        f"Ограничения: {bp.instructions.constraints}\n"
        f"Выходные поля: {[f.name for f in bp.io.fields]}\n"
        f"Критерии успеха: {bp.io.success_criteria}\n"
        f"Известные неоднозначности: {bp.instructions.ambiguities}\n"
        f"Верните не более {limit} вопросов."
    )
    data = llm.structured(
        task="clarify.questions", system=system, user=user, schema=_QUESTIONS_SCHEMA,
        context={"ambiguities": bp.instructions.ambiguities, "missing": bp.instructions.missing_data},
    )
    return list(data.get("questions", []))[:limit]


def _clip(s: str, n: int = 70) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 1] + "…"
