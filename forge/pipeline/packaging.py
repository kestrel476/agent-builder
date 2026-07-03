"""Этап 4 — финальная сборка.

Создаёт артефакты, которые нужны инженеру, чтобы подхватить агента: читаемую
спецификацию, отчёт о передаче, честно сообщающий о том, что осталось
непроверенным, обновлённый исполняемый комплект и запись в журнале изменений.
Комплект пересобирается, чтобы включить дополнения к промпту, полученные при
доработке.
"""

from __future__ import annotations

from ..generate import synthesize_bundle
from ..model import AgentBlueprint, Stage
from ..store import Workspace


def package(bp: AgentBlueprint, ws: Workspace, last_report=None) -> list:
    written = synthesize_bundle(bp, ws)  # комплект отражает актуальный промпт/контракт
    ws.spec_path.write_text(render_spec(bp), encoding="utf-8")
    (ws.root / "HANDOFF.md").write_text(render_handoff(bp, last_report), encoding="utf-8")
    bp.stage = Stage.PACKAGED
    bp.record("package", f"v{bp.version} упакована")
    ws.append_changelog(f"упакована v{bp.version} ({bp.archetype}); "
                        f"допущений: {len(bp.assumptions)}, открытых замечаний: {len(bp.open_issues)}")
    return written


# --------------------------------------------------------------------------- #
def render_spec(bp: AgentBlueprint) -> str:
    L: list[str] = [f"# {bp.name} — Спецификация", ""]
    L += [f"- **Идентификатор:** `{bp.slug}`",
          f"- **Архетип:** {bp.archetype}",
          f"- **Версия:** {bp.version}",
          f"- **Этап:** {bp.stage.value}",
          f"- **Возможности:** {', '.join(c.value for c in bp.capabilities)}", ""]
    L += ["## Цель", bp.goal or "_(не задано)_", ""]

    L += ["## Входные данные"]
    if bp.io.inputs:
        L += ["| имя | тип | обязателен | форматы |", "|---|---|---|---|"]
        for i in bp.io.inputs:
            L.append(f"| {i.name} | {i.kind} | {'да' if i.required else 'нет'} | {', '.join(i.formats) or '-'} |")
    else:
        L.append("_(не заданы)_")
    L.append("")

    L += ["## Контракт выхода", f"Тип: **{bp.io.output_kind.value}**", ""]
    if bp.io.fields:
        L += ["| поле | тип | обязательно | описание |", "|---|---|---|---|"]
        for f in bp.io.fields:
            L.append(f"| `{f.name}` | {f.type} | {'да' if f.required else 'нет'} | {f.description} |")
        L.append("")

    L += ["## Критерии успеха"] + [f"- {c}" for c in bp.io.success_criteria] + [""]
    L += ["## Политика ошибок / частичного результата", bp.io.error_definition, "", bp.io.partial_result_policy, ""]

    if bp.instructions.business_rules:
        L += ["## Бизнес-правила"] + [f"- {r}" for r in bp.instructions.business_rules] + [""]
    if bp.instructions.constraints:
        L += ["## Ограничения"] + [f"- {c}" for c in bp.instructions.constraints] + [""]
    if bp.assumptions:
        L += ["## Допущения (требуют проверки!)"]
        L += [f"- [{a.risk.value}] {a.statement} — _{a.because}_" for a in bp.assumptions] + [""]
    if bp.glossary:
        L += ["## Глоссарий"] + [f"- **{k}**: {v}" for k, v in bp.glossary.items()] + [""]
    return "\n".join(L)


def render_handoff(bp: AgentBlueprint, report=None) -> str:
    answered = sum(1 for q in bp.questions if q.answered)
    L: list[str] = [f"# Передача инженеру — {bp.name}", "",
                    "Этот агент был построен системой Legal Agent Forge через цикл "
                    "опрос → контракт → синтез → тестирование → доработка. Ниже — что надёжно "
                    "и что ещё требует внимания человека.", ""]
    L += ["## Статус",
          f"- Этап: **{bp.stage.value}**, версия **{bp.version}**",
          f"- Отвечено уточнений: {answered}/{len(bp.questions)}",
          f"- Записано допущений: {len(bp.assumptions)}",
          f"- Открытых замечаний: {len(bp.open_issues)}", ""]

    if report is not None:
        L += ["## Последний прогон тестов", f"- Пройдено: **{report.passed}/{report.total}**"]
        for r in report.results:
            L.append(f"  - {r.status.upper()}: {r.name} — {r.detail}")
        L.append("")

    unvalidated: list[str] = []
    if not bp.test_cases:
        unvalidated.append("Тест-кейсы не предоставлены — поведение не проверено.")
    if report is not None and report.failed:
        unvalidated.append(f"Кейсов всё ещё падает: {len(report.failed)}.")
    for a in bp.assumptions:
        if a.confirmed is not True:
            unvalidated.append(f"Неподтверждённое допущение: {a.statement}")
    for i in bp.open_issues:
        unvalidated.append(f"Открытое замечание ({i.severity.value}): {i.message}")
    L += ["## ⚠ Не проверено / требует внимания"]
    L += [f"- {u}" for u in unvalidated] if unvalidated else ["- Открытых вопросов нет."]
    L += ["", "## Как запустить", "```bash",
          "python agent/agent.py path/to/document.pdf --offline --debug", "```", ""]
    L += ["## История"] + [f"- {e.seq}. [{e.stage.value}] {e.action}: {e.detail}" for e in bp.history]
    return "\n".join(L)
