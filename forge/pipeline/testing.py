"""Этап 3 — тестирование, диагностика, доработка (инженерный цикл, а не один проход).

Прогоняет сгенерированного агента по кейсам пользователя, сравнивает фактический
результат с ожидаемым с учётом нормализации, удобной для юридических данных, и при
сбое просит «мозг» диагностировать первопричину и предложить конкретное
исправление, которое возвращается в чертёж. Цикл повторяется, пока набор тестов не
станет зелёным или пока прогресс не прекратится.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..integrations import DocumentReader, ForgeLLM
from ..model import AgentBlueprint, Stage, TestCase
from ..runtime import AgentInput, AgentSpec, run_agent
from ..store import Workspace


@dataclass
class CaseResult:
    id: str
    name: str
    status: str           # pass | fail | error | skip (статус кейса)
    detail: str = ""
    diffs: list[dict] = field(default_factory=list)
    run_status: str = ""
    output: Any = None


@dataclass
class TestReport:
    results: list[CaseResult] = field(default_factory=list)
    run_dir: Path | None = None

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.status == "pass")

    @property
    def failed(self) -> list[CaseResult]:
        return [r for r in self.results if r.status in ("fail", "error")]

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def green(self) -> bool:
        return bool(self.results) and not self.failed

    def rows(self) -> list[dict]:
        return [{"name": r.name, "status": r.status, "detail": r.detail} for r in self.results]


# --------------------------------------------------------------------------- #
# Прогон
# --------------------------------------------------------------------------- #
def run_tests(bp: AgentBlueprint, ws: Workspace, llm: ForgeLLM,
              reader: DocumentReader | None = None, *, only: str | None = None) -> TestReport:
    if not ws.agent_dir.joinpath("agent.yaml").is_file():
        raise FileNotFoundError("Нет комплекта агента. Сначала выполните `forge build`.")
    spec = AgentSpec.load(ws.agent_dir)
    reader = reader or DocumentReader()
    report = TestReport()

    cases = [c for c in bp.test_cases if only in (None, c.id, c.name)]
    for tc in cases:
        report.results.append(_run_case(spec, tc, ws, llm, reader))

    report.run_dir = _persist(ws, report)
    bp.stage = Stage.TESTED
    bp.record("test", f"пройдено {report.passed}/{report.total}; прогон={report.run_dir.name if report.run_dir else '-'}")
    return report


def _run_case(spec: AgentSpec, tc: TestCase, ws: Workspace, llm: ForgeLLM,
              reader: DocumentReader) -> CaseResult:
    inp = _build_input(tc, ws)
    if inp is None:
        return CaseResult(tc.id, tc.name, "skip", "вход не предоставлен")
    res = run_agent(spec, inp, llm=llm, reader=reader)
    if res.status == "error":
        return CaseResult(tc.id, tc.name, "error", "; ".join(res.errors) or "ошибка времени выполнения",
                          run_status=res.status, output=res.output)

    diffs: list[dict] = []
    detail_parts: list[str] = [f"прогон={res.status}"]
    if tc.expected is not None:
        diffs = diff_objects(tc.expected, res.output if isinstance(res.output, dict) else {})
    if tc.expected_text is not None:
        if _norm(tc.expected_text) != _norm(str(res.output)):
            diffs.append({"field": "<text>", "expected": tc.expected_text, "actual": str(res.output)[:200]})
    missing_substr = [s for s in tc.must_contain if s.lower() not in str(res.output).lower()]
    if missing_substr:
        diffs.append({"field": "<must_contain>", "expected": missing_substr, "actual": "absent"})

    metric_fail = False
    if tc.metric:
        ok, summary, diff = _eval_metric(tc.metric, res.output)
        detail_parts.append(summary)
        if not ok:
            metric_fail = True
            diffs.append(diff)

    if diffs:
        detail_parts.append(f"несовпадений: {len(diffs)}")
        status = "fail"
    elif metric_fail:
        status = "fail"
    elif tc.expected is None and tc.expected_text is None and not tc.must_contain and not tc.metric:
        status = "pass"  # дымовой тест: отработал без ошибок
        detail_parts.append("дымовой тест ок (без ожиданий)")
    else:
        status = "pass"
    return CaseResult(tc.id, tc.name, status, "; ".join(detail_parts), diffs=diffs,
                      run_status=res.status, output=res.output)


def _eval_metric(metric: dict, output) -> tuple[bool, str, dict]:
    """Считает precision/recall/F1 для list-поля по ключу и сверяет с порогами."""
    field_name = metric.get("field", "")
    key = metric.get("key")
    items = output.get(field_name, []) if isinstance(output, dict) else []
    if key:
        got = {str(it.get(key)) for it in items if isinstance(it, dict) and it.get(key) is not None}
    else:
        got = {str(it) for it in items}
    expected = {str(e) for e in (metric.get("expected") or [])}
    tp = expected & got
    precision = len(tp) / len(got) if got else 0.0
    recall = len(tp) / len(expected) if expected else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    min_r = float(metric.get("min_recall", 1.0))
    min_p = float(metric.get("min_precision", 0.0))
    ok = recall >= min_r and precision >= min_p
    summary = (f"{field_name}: P={precision:.2f} R={recall:.2f} F1={f1:.2f} "
               f"(найдено {len(tp)}/{len(expected)} ожид., всего {len(got)})")
    diff = {"field": field_name, "expected": sorted(expected), "actual": sorted(got),
            "metric": {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}}
    return ok, summary, diff


def _resolve_input_path(name: str, ws: Workspace) -> str:
    p = Path(name)
    if not p.is_absolute():
        p = ws.inputs_dir / name
        if not p.exists():
            p = ws.root / name
    return str(p)


def _build_input(tc: TestCase, ws: Workspace) -> AgentInput | None:
    if tc.input_files:  # несколько документов (compare/package)
        return AgentInput(files=[_resolve_input_path(f, ws) for f in tc.input_files])
    if tc.input_file:
        return AgentInput(files=[_resolve_input_path(tc.input_file, ws)])
    if tc.input_json is not None:
        return AgentInput(json=tc.input_json, question=tc.input_text)
    if tc.input_text is not None:
        return AgentInput(text=tc.input_text, question=tc.input_text)
    return None


# --------------------------------------------------------------------------- #
# Сравнение (нормализация, удобная для юридических данных)
# --------------------------------------------------------------------------- #
def diff_objects(expected: dict, actual: dict) -> list[dict]:
    diffs = []
    for key, exp in expected.items():
        act = actual.get(key)
        if _norm(exp) != _norm(act):
            diffs.append({"field": key, "expected": exp, "actual": act})
    return diffs


def _norm(v: Any) -> Any:
    if isinstance(v, str):
        return " ".join(v.strip().lower().split())
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return round(float(v), 4)
    if isinstance(v, list):
        return sorted(json.dumps(_norm(x), sort_keys=True, ensure_ascii=False) for x in v)
    if isinstance(v, dict):
        return {k: _norm(v[k]) for k in sorted(v)}
    return v


# --------------------------------------------------------------------------- #
# Диагностика + доработка
# --------------------------------------------------------------------------- #
_DIAG_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "root_cause": {"type": "string"},
        "fix_suggestion": {"type": "string"},
        "blueprint_patch": {"type": "object"},
    },
    "required": ["root_cause", "fix_suggestion"],
}


def diagnose(bp: AgentBlueprint, report: TestReport, llm: ForgeLLM) -> list[dict]:
    diagnoses: list[dict] = []
    for case in report.failed:
        system = (
            "Ты отлаживаешь падающий тест юридического извлечения. Объясни первопричину "
            "несовпадения и предложи конкретное минимальное исправление (уточнение промпта, "
            "изменение схемы или исправленное ожидание). Верни небольшой blueprint_patch."
        )
        user = (
            f"Цель: {bp.goal}\nКейс: {case.name}\nСтатус прогона: {case.run_status}\n"
            f"Несовпадения: {json.dumps(case.diffs, ensure_ascii=False)[:1500]}"
        )
        d = llm.structured(task="test.diagnose", system=system, user=user, schema=_DIAG_SCHEMA,
                           context={"diffs": case.diffs})
        d["case"] = case.name
        diagnoses.append(d)
    return diagnoses


_REFINE_SCHEMA = {
    "type": "object",
    "properties": {"prompt_addendum": {"type": "string"}, "notes": {"type": "string"}},
    "required": ["prompt_addendum"],
}


def refine(bp: AgentBlueprint, diagnoses: list[dict], llm: ForgeLLM) -> str:
    if not diagnoses:
        return "нечего дорабатывать"
    system = (
        "По диагнозам падающих кейсов сформируй единственное краткое дополнение к "
        "системному промпту агента, которое устраняет систематические проблемы, не "
        "переобучаясь на отдельные примеры."
    )
    user = "Диагнозы:\n" + json.dumps(diagnoses, ensure_ascii=False)[:3000]
    data = llm.structured(task="refine.propose", system=system, user=user, schema=_REFINE_SCHEMA,
                          context={"diagnoses": diagnoses})
    addendum = (data.get("prompt_addendum") or "").strip()
    if addendum and addendum not in bp.prompt_addenda:
        bp.prompt_addenda.append(addendum)
        bp.version += 1
        bp.record("refine", f"v{bp.version}: добавлено дополнение к промпту ({len(addendum)} симв.)")
    return addendum or "дополнение не сформировано"


def _persist(ws: Workspace, report: TestReport) -> Path:
    d = ws.new_run_dir()
    payload = {
        "passed": report.passed,
        "total": report.total,
        "results": [
            {"id": r.id, "name": r.name, "status": r.status, "detail": r.detail,
             "diffs": r.diffs, "run_status": r.run_status}
            for r in report.results
        ],
    }
    (d / "result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return d
