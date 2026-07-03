"""Этап 1 — зафиксировать явный, тестируемый контракт входов/выходов.

Сводит воедино значения архетипа по умолчанию, структурированные инструкции и
всё, что установили ответы на уточнения, в: объявленные входы (обязательные/
необязательные + форматы), схему выхода (типизированные поля, флаги
обязательности), критерии успеха и явную политику ошибок / частичного результата /
низкой уверенности.
"""

from __future__ import annotations

from ..archetypes import get as get_archetype
from ..integrations import ForgeLLM
from ..model import (
    AgentBlueprint,
    FieldSpec,
    InputSpec,
    OutputKind,
    Stage,
)

_FIELDS_SCHEMA = {
    "type": "object",
    "properties": {
        "fields": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string"},
                    "required": {"type": "boolean"},
                    "description": {"type": "string"},
                },
                "required": ["name"],
            },
        }
    },
    "required": ["fields"],
}


def build_contract(bp: AgentBlueprint, llm: ForgeLLM) -> None:
    arch = get_archetype(bp.archetype)
    arch.seed_contract(bp)

    _ensure_fields(bp, llm, arch)
    _ensure_categories(bp, llm)
    _ensure_inputs(bp, arch)
    _ensure_success_criteria(bp)
    _ensure_error_policy(bp)
    _absorb_answers(bp)
    _reconcile_issues(bp)

    if not bp.blocking_issues:
        bp.stage = Stage.CONTRACTED
    bp.record("contract", f"полей: {len(bp.io.fields)}, входов: {len(bp.io.inputs)}")


def _reconcile_issues(bp: AgentBlueprint) -> None:
    """Закрыть замечания, которые зафиксированный контракт уже удовлетворил.

    Когда этап контракта сам заполнил входы / критерии успеха / политику ошибок /
    поля, соответствующие замечания критиков больше не актуальны — снимаем их
    (и помечаем связанные вопросы), чтобы не блокировать сборку тем, что уже
    решено. Семантические вопросы (список полей, нормализация) остаются открытыми.
    """
    from ..model import IssueStatus

    io = bp.io
    for issue in bp.issues:
        if issue.status != IssueStatus.OPEN:
            continue
        w = issue.where
        satisfied = (
            (w == "io.inputs" and bool(io.inputs))
            or (w == "io.success_criteria" and bool(io.success_criteria))
            or (w == "io.error_definition" and bool(io.error_definition.strip()))
            or (w == "io.fields" and "required" not in issue.message.lower() and bool(io.fields))
            or (w == "io.fields" and "required" in issue.message.lower()
                and any(f.required for f in io.fields))
        )
        if satisfied:
            issue.status = IssueStatus.RESOLVED
            for q in bp.questions:
                if q.issue_id == issue.id and not q.answered:
                    q.answer = "[авто] разрешено при фиксации контракта"


# --------------------------------------------------------------------------- #
def _ensure_fields(bp: AgentBlueprint, llm: ForgeLLM, arch) -> None:
    if bp.io.output_kind in (OutputKind.NEW_DOCUMENT, OutputKind.EDITED_DOCUMENT, OutputKind.CHAT):
        return  # у свободнотекстовых выходов нет схемы полей
    if arch.default_fields:
        return  # архетип задаёт фиксированную схему — не доинферим поля
    if len(bp.io.fields) >= 2:
        return
    system = (
        "Предложи поля выхода для агента юридической автоматизации. Каждому полю "
        "нужны имя (snake_case), JSON-тип, признак обязательности и однострочное "
        "описание. Предпочитай небольшую точную схему большой умозрительной."
    )
    user = (
        f"Цель: {bp.goal}\nАрхетип: {bp.archetype}\n"
        f"Упомянутые выходы: {bp.instructions.outputs}\n"
        f"Бизнес-правила: {bp.instructions.business_rules}\n"
        f"Существующие поля: {[f.name for f in bp.io.fields]}"
    )
    data = llm.structured(task="contract.output_schema", system=system, user=user,
                          schema=_FIELDS_SCHEMA, context={"raw": bp.instructions_raw})
    existing = {f.name for f in bp.io.fields}
    for spec in data.get("fields", []):
        name = (spec.get("name") or "").strip()
        if not name or name in existing:
            continue
        bp.io.fields.append(FieldSpec(
            name=name, type=spec.get("type", "string"),
            required=bool(spec.get("required", False)),
            description=spec.get("description", "")))
        existing.add(name)


_CATEGORIES_SCHEMA = {
    "type": "object",
    "properties": {"categories": {"type": "array", "items": {"type": "string"}}},
    "required": ["categories"],
}


def _ensure_categories(bp: AgentBlueprint, llm: ForgeLLM) -> None:
    """Для классификации — извлечь закрытый список категорий в ``label.enum``.

    Превращает категории из инструкции в перечисление, чтобы агент выбирал метку
    из фиксированного набора, а не выдумывал свою. Без категорий enum не ставится.
    """
    if bp.io.output_kind != OutputKind.CLASSIFICATION:
        return
    label = next((f for f in bp.io.fields if f.name == "label"), None)
    if label is None or label.enum:
        return
    system = (
        "Извлеки закрытый список категорий классификации из инструкции. Верни их "
        "точные краткие метки в нижнем регистре, включая запасную категорию для "
        "прочего, если она подразумевается. Только метки, без пояснений."
    )
    user = f"Цель: {bp.goal}\nИнструкции: {bp.instructions_raw}"
    data = llm.structured(task="contract.categories", system=system, user=user,
                          schema=_CATEGORIES_SCHEMA, context={"raw": bp.instructions_raw})
    cats = [c.strip() for c in data.get("categories", []) if isinstance(c, str) and c.strip()]
    if cats:
        label.enum = cats[:20]


def _ensure_inputs(bp: AgentBlueprint, arch) -> None:
    if bp.io.inputs:
        return
    if arch.runtime_strategy == "chat":
        bp.io.inputs = [
            InputSpec(name="question", kind="text", required=True, description="Вопрос пользователя."),
            InputSpec(name="knowledge", kind="document", required=False,
                      formats=["pdf", "docx", "txt"], description="Документы-основания."),
        ]
    elif arch.input_arity >= 2:
        bp.io.inputs = [
            InputSpec(name="document_a", kind="document", required=True,
                      formats=["pdf", "docx"], description="Базовый документ."),
            InputSpec(name="document_b", kind="document", required=True,
                      formats=["pdf", "docx"], description="Проверяемый документ."),
        ]
    else:
        bp.io.inputs = [InputSpec(
            name="document", kind="document", required=True,
            formats=["pdf", "docx", "txt", "eml", "png"],
            description="Юридический документ для обработки.")]


def _ensure_success_criteria(bp: AgentBlueprint) -> None:
    if bp.io.success_criteria:
        return
    req = [f.name for f in bp.io.fields if f.required]
    crit: list[str] = []
    if req:
        crit.append(f"Все обязательные поля присутствуют и непусты: {', '.join(req)}.")
    crit.append("Значения полей прослеживаются до исходного документа (без выдумок).")
    if bp.io.confidence_required:
        crit.append("Результат сопровождается калиброванной оценкой уверенности.")
    bp.io.success_criteria = crit


def _ensure_error_policy(bp: AgentBlueprint) -> None:
    if not bp.io.error_definition:
        bp.io.error_definition = (
            "ERROR: вход нечитаем или ни одно обязательное поле невозможно определить. "
            "PARTIAL: часть обязательных полей отсутствует (выводятся как null). "
            "LOW_CONFIDENCE: результат ниже порога уверенности."
        )
    if not bp.io.partial_result_policy:
        bp.io.partial_result_policy = (
            "Выводи каждое поле, которое можно определить; неопределимые поля задавай как null; "
            "никогда не отбрасывай оболочку результата. Отмечай частичные результаты в `status`."
        )


def _absorb_answers(bp: AgentBlueprint) -> None:
    """Свести отвеченные уточнения в проверяемые предметные заметки."""
    qa = [(q.text, q.answer) for q in bp.questions if q.answered]
    if not qa:
        return
    lines = ["Уточнения, зафиксированные при сборе требований:"]
    lines += [f"- В: {t}\n  О: {a}" for t, a in qa]
    note = "\n".join(lines)
    if note not in bp.domain_notes:
        bp.domain_notes = (bp.domain_notes + "\n\n" + note).strip() if bp.domain_notes else note
