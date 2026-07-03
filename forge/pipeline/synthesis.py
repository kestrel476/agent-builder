"""Этап 2 — синтез агента.

Собирает системный промпт из *разрешённого* чертежа (цель, правила, контракт,
политика ошибок, записанные допущения) и формирует исполняемый комплект.
Генерация происходит только после уточнения, потому что промпт хорош ровно
настолько, насколько хороша стоящая за ним спецификация.
"""

from __future__ import annotations

from ..generate import synthesize_bundle
from ..integrations import ForgeLLM
from ..model import AgentBlueprint, Stage
from ..store import Workspace

_PROMPT_SYSTEM = (
    "Напиши точный системный промпт для специализированного агента юридической "
    "автоматизации. Он должен: формулировать единственную цель; перечислять "
    "бизнес-правила; описывать точный контракт выхода и политику ошибок/частичного "
    "результата/низкой уверенности; запрещать выдумывание; и предписывать агенту "
    "копировать значения дословно там, где это требуется. Будь кратким и предметным — "
    "этот промпт управляет агентом в продакшене."
)


def synthesize(bp: AgentBlueprint, ws: Workspace, llm: ForgeLLM) -> list:
    bp.system_prompt = _generate_prompt(bp, llm)
    written = synthesize_bundle(bp, ws)
    bp.stage = Stage.BUILT
    bp.record("synthesize", f"системный промпт ({len(bp.system_prompt)} симв.) + файлов комплекта: {len(written)}")
    return written


def _generate_prompt(bp: AgentBlueprint, llm: ForgeLLM) -> str:
    rules = bp.instructions.business_rules or [f.description for f in bp.io.fields if f.description]
    assumptions = [a.statement for a in bp.assumptions]
    user = (
        f"Цель: {bp.goal}\nАрхетип: {bp.archetype}\n"
        f"Поля выхода: {[(f.name, f.type, f.required) for f in bp.io.fields]}\n"
        f"Критерии успеха: {bp.io.success_criteria}\n"
        f"Политика ошибок: {bp.io.error_definition}\n"
        f"Бизнес-правила: {rules}\n"
        f"Ограничения: {bp.instructions.constraints}\n"
        f"Записанные допущения (соблюдай их): {assumptions}\n"
        f"Глоссарий: {bp.glossary}"
    )
    text = llm.text(task="synthesis.system_prompt", system=_PROMPT_SYSTEM, user=user,
                    context={"goal": bp.goal, "rules": rules}, temperature=0.2)
    if text.strip():
        return text
    # Офлайн / пусто: детерминированный промпт, собранный из контракта.
    rules_block = "\n".join(f"- {r}" for r in rules) or "- Точно следуй схеме выхода."
    asm_block = "\n".join(f"- {a}" for a in assumptions)
    return (
        f"Ты — специализированный агент юридической автоматизации.\nЦель: {bp.goal}\n\n"
        f"Бизнес-правила:\n{rules_block}\n\n"
        f"Контракт выхода: верни объект с полями "
        f"{[f.name for f in bp.io.fields]}; обязательные: {[f.name for f in bp.io.fields if f.required]}.\n"
        f"Политика ошибок/частичного результата: {bp.io.error_definition}\n\n"
        + (f"Рабочие допущения:\n{asm_block}\n\n" if asm_block else "")
        + "Работай только по предоставленному документу. Если обязательное значение отсутствует, "
          "задай его как null и понизь свою уверенность. Никогда не выдумывай факты."
    )
