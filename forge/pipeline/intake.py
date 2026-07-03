"""Этап 0 — приём инструкций.

Свободные юридические инструкции нормализуются в структурированное промежуточное
представление, *размеченное* по уровням (цель / входы / выходы / правила /
ограничения / неоднозначности / отсутствующее / риски). Важно: смысл не
выдумывается — неоднозначные или отсутствующие элементы фиксируются как таковые,
чтобы модуль уточнения мог их поднять, а не разрешались молча.
"""

from __future__ import annotations

from ..archetypes import get as get_archetype
from ..integrations import ForgeLLM
from ..model import AgentBlueprint, Stage, StructuredInstructions

_INSTR_SCHEMA = {
    "type": "object",
    "properties": {
        "goal": {"type": "string"},
        "scope_in": {"type": "array", "items": {"type": "string"}},
        "scope_out": {"type": "array", "items": {"type": "string"}},
        "inputs": {"type": "array", "items": {"type": "string"}},
        "outputs": {"type": "array", "items": {"type": "string"}},
        "business_rules": {"type": "array", "items": {"type": "string"}},
        "constraints": {"type": "array", "items": {"type": "string"}},
        "definitions": {"type": "array", "items": {"type": "string"}},
        "ambiguities": {"type": "array", "items": {"type": "string"}},
        "missing_data": {"type": "array", "items": {"type": "string"}},
        "interpretation_risks": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["goal"],
}

_ARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "archetype": {"type": "string"},
        "confidence": {"type": "number"},
        "rationale": {"type": "string"},
    },
    "required": ["archetype"],
}

_INTAKE_SYSTEM = (
    "Ты нормализуешь свободные юридические инструкции в точную спецификацию. "
    "Классифицируй каждое утверждение по уровню: цель, что входит/не входит в рамки, "
    "входы, выходы, бизнес-правила, ограничения, определения. Отдельно перечисли "
    "неоднозначности, отсутствующие данные и риски интерпретации. НИКОГДА не угадывай "
    "смысл — если что-то неясно или отсутствует, занеси это в ambiguities/missing_data, "
    "не додумывай."
)


def run_intake(bp: AgentBlueprint, llm: ForgeLLM) -> StructuredInstructions:
    raw = bp.instructions_raw.strip()
    user = f"Исходные инструкции:\n\n{raw}\n\nВерни структурированную спецификацию."
    data = llm.structured(task="intake.structure", system=_INTAKE_SYSTEM, user=user,
                          schema=_INSTR_SCHEMA, context={"raw": raw})
    # Оставляем только известные ключи; pydantic игнорирует лишние, но укажем явно.
    known = StructuredInstructions.model_fields.keys()
    bp.instructions = StructuredInstructions(**{k: data.get(k) for k in known if data.get(k) is not None})
    if not bp.goal:
        bp.goal = bp.instructions.goal
    get_archetype(bp.archetype).seed_contract(bp)
    bp.stage = Stage.INTAKE
    bp.record("intake", f"цель='{bp.goal[:60]}'; неоднозначностей: {len(bp.instructions.ambiguities)}, "
                        f"отсутствующих элементов: {len(bp.instructions.missing_data)}")
    return bp.instructions


def detect_archetype(raw: str, llm: ForgeLLM) -> tuple[str, float, str]:
    from ..archetypes import keys as arch_keys
    system = (
        "Выбери единственный наиболее подходящий архетип агента для описанной "
        f"юридической задачи. Выбери ровно один из: {', '.join(arch_keys())}."
    )
    data = llm.structured(task="intake.detect_archetype", system=system,
                          user=f"Задача:\n{raw}", schema=_ARCH_SCHEMA, context={"raw": raw})
    key = data.get("archetype", "json_extraction")
    if key not in arch_keys():
        key = "json_extraction"
    return key, float(data.get("confidence", 0.5)), data.get("rationale", "")
