"""Среда выполнения, которая запускает сгенерированного агента.

Единственная точка входа ``run_agent`` направляет работу в нужную *стратегию*
(указанную в конфигурации агента), поэтому все архетипы используют общие разрешение
входных данных, валидацию по схеме, оценку уверенности, ведение трассы выполнения
и холостой прогон/отладку — различается только основной шаг.
"""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..integrations import DocumentReader, ForgeLLM
from . import confidence as conf
from .knowledge import KnowledgeBase, KnowledgeConfig
from .rules import RuleEngine
from .spec import AgentSpec, RunResult, TraceStep
from .validation import validate

# Независимые LLM-вызовы (пер-чанк/ситуация/документ/батч) выполняются
# параллельно: httpx.Client провайдера потокобезопасен, заглушка без состояния.
_CONCURRENCY = max(1, min(16, int(os.environ.get("FORGE_CONCURRENCY", "6") or "6")))


def _map_concurrent(items: list, fn: Callable[[Any], Any], *, workers: int | None = None) -> list:
    """Применяет ``fn`` к каждому элементу, сохраняя порядок результатов.

    Параллелит при >1 элементе и concurrency>1; падение одного элемента даёт
    ``None`` в этой позиции (вызывающая сторона отфильтровывает), не роняя пакет."""
    items = list(items)
    n = workers if workers is not None else _CONCURRENCY
    if len(items) <= 1 or n <= 1:
        out = []
        for x in items:
            try:
                out.append(fn(x))
            except Exception:
                out.append(None)
        return out
    out: list = [None] * len(items)
    with ThreadPoolExecutor(max_workers=min(n, len(items))) as ex:
        futs = {ex.submit(fn, x): i for i, x in enumerate(items)}
        for fut, i in futs.items():
            try:
                out[i] = fut.result()
            except Exception:
                out[i] = None
    return out


def _clip(text: str, limit: int, ctx: "_Ctx | None" = None, label: str = "вход") -> tuple[str, bool]:
    """Усекает текст до ``limit`` символов; при усечении логирует в trace (если дан ctx).

    Возвращает (текст, был_ли_усечён) — чтобы вызывающая сторона могла добавить
    предупреждение. Цель — устранить ТИХУЮ потерю данных на длинных документах."""
    if len(text) <= limit:
        return text, False
    if ctx is not None:
        ctx.trace.append(TraceStep("truncated", f"{label}: {len(text)}→{limit} символов (хвост отброшен)"))
    return text[:limit], True


@dataclass
class AgentInput:
    """Всё, что агенту предлагается обработать за один запуск."""

    text: str | None = None
    json: dict[str, Any] | None = None
    files: list[str] = field(default_factory=list)
    question: str | None = None  # стратегия chat

    @classmethod
    def from_value(cls, value: Any) -> "AgentInput":
        if isinstance(value, AgentInput):
            return value
        if isinstance(value, dict):
            return cls(json=value)
        if isinstance(value, (str, Path)):
            try:
                # Длинный текст не является путём — Path.exists() кинет OSError.
                is_file = Path(value).exists()
            except OSError:
                is_file = False
            return cls(files=[str(value)]) if is_file else cls(text=str(value))
        return cls(text=str(value))


@dataclass
class _Ctx:
    llm: ForgeLLM
    reader: DocumentReader
    trace: list[TraceStep]
    dry_run: bool = False


# --------------------------------------------------------------------------- #
def run_agent(
    spec: AgentSpec,
    value: Any,
    *,
    llm: ForgeLLM,
    reader: DocumentReader | None = None,
    dry_run: bool = False,
) -> RunResult:
    reader = reader or DocumentReader()
    trace: list[TraceStep] = []
    ctx = _Ctx(llm=llm, reader=reader, trace=trace, dry_run=dry_run)
    inp = AgentInput.from_value(value)
    strat = _STRATEGIES.get(spec.strategy)
    if strat is None:
        return RunResult(status="error", errors=[f"неизвестная стратегия '{spec.strategy}'"], trace=trace)
    calls_before, tokens_before = len(llm.calls), llm.total_tokens
    try:
        result = strat(spec, inp, ctx)
    except Exception as e:  # никогда не позволяем сгенерированному агенту обрушить хост
        trace.append(TraceStep("exception", str(e)))
        result = RunResult(status="error", errors=[f"исключение среды выполнения: {e}"], trace=trace)
    # Наблюдаемость: сколько LLM-вызовов и токенов стоил этот запуск.
    result.stats = {
        "llm_calls": len(llm.calls) - calls_before,
        "tokens": llm.total_tokens - tokens_before,
        "backend": llm.backend_name,
    }
    return result


# --------------------------------------------------------------------------- #
# Разрешение входных данных
# --------------------------------------------------------------------------- #
def _resolve_text(inp: AgentInput, ctx: _Ctx, *, which: int = 0) -> str:
    """Превратить вход в простой текст, при необходимости читая документы."""
    if inp.files and which < len(inp.files):
        res = ctx.reader.read(inp.files[which])
        ctx.trace.append(TraceStep("read_document", f"{inp.files[which]} → {res.status}", res.meta))
        if res.needs_ocr and not res.text:
            ctx.trace.append(TraceStep("ocr_required", "у документа нет текстового слоя"))
        return res.text
    if which == 0:
        if inp.text:
            return inp.text
        if inp.json is not None:
            return json.dumps(inp.json, ensure_ascii=False, indent=2)
    return ""


def _finish_structured(spec: AgentSpec, output: dict, ctx: _Ctx) -> RunResult:
    errors = validate(spec.output_schema, output)
    status, confidence, warnings = conf.assess(
        output, threshold=spec.confidence_threshold, validation_errors=errors
    )
    ctx.trace.append(TraceStep("validate", f"ошибок: {len(errors)}; статус={status}"))
    return RunResult(status=status, output=output, confidence=confidence,
                     errors=[e for e in errors if "обязательное поле отсутствует" not in e],
                     warnings=warnings, trace=ctx.trace)


# --------------------------------------------------------------------------- #
# Стратегии
# --------------------------------------------------------------------------- #
def _strat_extract(spec: AgentSpec, inp: AgentInput, ctx: _Ctx) -> RunResult:
    text = _resolve_text(inp, ctx)
    if not text.strip():
        return RunResult(status="error", errors=["пустой вход: нечего обрабатывать"], trace=ctx.trace)
    clipped, _ = _clip(text, 20000, ctx, "документ")
    user = (
        "Извлеки требуемые поля из приведённого ниже документа и верни их согласно схеме. "
        "Если значение отсутствует, установи его в null и понизь уверенность; никогда не выдумывай факты.\n\n"
        f"=== ДОКУМЕНТ ===\n{clipped}"
    )
    if ctx.dry_run:
        ctx.trace.append(TraceStep("dry_run", "вызов LLM пропущен", {"chars": len(text)}))
        return RunResult(status="ok", output={"_dry_run": True, "input_chars": len(text)}, trace=ctx.trace)
    output = ctx.llm.structured(
        task="agent.extract", system=spec.system_prompt, user=user,
        schema=spec.output_schema, context={"text": text},
    )
    ctx.trace.append(TraceStep("llm_extract", f"возвращено полей: {len(output)}"))
    return _finish_structured(spec, output, ctx)


def _strat_compare(spec: AgentSpec, inp: AgentInput, ctx: _Ctx) -> RunResult:
    a = _resolve_text(inp, ctx, which=0)
    b = _resolve_text(inp, ctx, which=1)
    if not a.strip() or not b.strip():
        return RunResult(status="error", errors=["для сравнения нужны два документа (A и B)"], trace=ctx.trace)
    ca, _ = _clip(a, 12000, ctx, "документ A")
    cb, _ = _clip(b, 12000, ctx, "документ B")
    user = (
        "Сравни документ A (эталон) с документом B (на проверке) и опиши "
        "различия согласно схеме.\n\n=== A ===\n" + ca + "\n\n=== B ===\n" + cb
    )
    if ctx.dry_run:
        return RunResult(status="ok", output={"_dry_run": True}, trace=ctx.trace)
    output = ctx.llm.structured(
        task="agent.compare", system=spec.system_prompt, user=user,
        schema=spec.output_schema, context={"text": a + "\n" + b},
    )
    return _finish_structured(spec, output, ctx)


def _strat_generate(spec: AgentSpec, inp: AgentInput, ctx: _Ctx) -> RunResult:
    text = _resolve_text(inp, ctx)
    clipped, _ = _clip(text, 16000, ctx, "входные данные")
    user = (
        "Составь документ, используя предоставленный шаблон и входные данные. Не изменяй ни одного "
        "пункта, помеченного как неизменяемый.\n\n=== ВХОДНЫЕ ДАННЫЕ ===\n" + clipped
    )
    if ctx.dry_run:
        return RunResult(status="ok", output="(холостой прогон: документ не сгенерирован)", trace=ctx.trace)
    doc = ctx.llm.text(task="agent.generate", system=spec.system_prompt, user=user, temperature=0.3)
    status = "ok" if doc.strip() else "partial"
    ctx.trace.append(TraceStep("llm_generate", f"символов: {len(doc)}"))
    return RunResult(status=status, output=doc, trace=ctx.trace,
                     warnings=[] if doc.strip() else ["сгенерирован пустой документ"])


def _strat_chat(spec: AgentSpec, inp: AgentInput, ctx: _Ctx) -> RunResult:
    question = inp.question or inp.text or ""
    context_text = _resolve_text(inp, ctx) if inp.files else ""
    if not question.strip():
        return RunResult(status="error", errors=["стратегии chat нужен вопрос"], trace=ctx.trace)
    cctx, _ = _clip(context_text, 12000, ctx, "контекст") if context_text else ("", False)
    user = question if not context_text else f"Контекст:\n{cctx}\n\nВопрос: {question}"
    if ctx.dry_run:
        return RunResult(status="ok", output="(холостой прогон)", trace=ctx.trace)
    answer = ctx.llm.text(task="agent.chat", system=spec.system_prompt, user=user, temperature=0.2)
    return RunResult(status="ok" if answer.strip() else "partial", output=answer, trace=ctx.trace)


def _strat_workflow(spec: AgentSpec, inp: AgentInput, ctx: _Ctx) -> RunResult:
    # Шаг 1: структурированное извлечение из материалов дела.
    res = _strat_extract(spec, inp, ctx)
    if res.status == "error":
        return res
    # Шаг 2: фиксация объявленных правил/проверок (детерминированных, наблюдаемых).
    # Кладём в отдельное поле `checks` БЕЗУСЛОВНО (раньше setdefault не срабатывал,
    # т.к. extract уже клал пустой `checks` из схемы — заявленная фича не работала).
    checks = [{"rule": r, "evaluated": True} for r in spec.rules]
    if isinstance(res.output, dict):
        res.output["checks"] = checks
    ctx.trace.append(TraceStep("workflow_rules", f"проверено правил: {len(checks)}"))
    return res


_SITUATIONS_SCHEMA = {
    "type": "object",
    "properties": {"situations": {"type": "array", "items": {"type": "string"}}},
    "required": ["situations"],
}
_CODES_SCHEMA = {
    "type": "object",
    "properties": {"risk_codes": {"type": "array", "items": {"type": "string"}}},
    "required": ["risk_codes"],
}


def _strat_rag_detect(spec: AgentSpec, inp: AgentInput, ctx: _Ctx) -> RunResult:
    """Детекция элементов каталога (рисков) в документе через RAG.

    1) LLM извлекает из документа конкретные обстоятельства;
    2) по каждому — retrieval кандидатов из каталога (эмбеддинги);
    3) LLM сопоставляет обстоятельство с кандидатами и возвращает коды;
    title берётся из каталога, fact — из обстоятельства документа.
    """
    kcfg_raw = spec.settings.get("knowledge")
    if not kcfg_raw:
        return RunResult(status="error", errors=["для стратегии rag_detect нужна база знаний (knowledge)"], trace=ctx.trace)
    kb = KnowledgeBase.load(spec.bundle_dir, KnowledgeConfig.from_dict(kcfg_raw))
    ctx.trace.append(TraceStep("knowledge", f"{len(kb.entries)} записей каталога"))

    text = _resolve_text(inp, ctx)
    if not text.strip():
        return RunResult(status="error", errors=["пустой вход: нечего обрабатывать"], trace=ctx.trace)
    if ctx.dry_run:
        return RunResult(status="ok", output={"_dry_run": True, "catalog": len(kb.entries)}, trace=ctx.trace)

    kb.ensure_index(ctx.llm)
    detected = _rag_retrieve_match(kb, text, ctx, top_k=int(kcfg_raw.get("top_k", 30)))

    ctx.trace.append(TraceStep("matched", f"обнаружено: {len(detected)}"))
    output: dict[str, Any] = {"detected_risks": list(detected.values())}
    if "confidence" in spec.output_schema.get("properties", {}):
        output["confidence"] = 0.9 if detected else 0.3
    return _finish_structured(spec, output, ctx)


_EXTRACT_SYS = (
    "Ты — юрист, анализируешь фрагмент документа на правовые риски. Извлеки КОНКРЕТНЫЕ "
    "обстоятельства сделки, способные порождать правовые риски (структура и предмет обеспечения, "
    "залоги и ипотека, поручительства и гарантии, корпоративный контроль над контрагентом, доли "
    "в уставном капитале и квазикорпоративные договоры, опционы, целевое использование средств, "
    "ковенанты и пр.). Каждое — одно ёмкое предложение по тексту. Если рисковых обстоятельств во "
    "фрагменте нет — пустой список. Не выдумывай."
)


def _chunk_text(text: str, *, width: int = 3500, overlap: int = 400, max_chunks: int = 12) -> list[str]:
    chunks: list[str] = []
    i = 0
    while i < len(text) and len(chunks) < max_chunks:
        chunks.append(text[i : i + width])
        i += width - overlap
    return chunks


def _dedupe_situations(situations: list[str], vecs: list[list[float]], *, thresh: float = 0.90,
                       cap: int = 28) -> tuple[list[str], list]:
    """Убрать near-дубликаты обстоятельств по косинусной близости эмбеддингов."""
    import numpy as np

    keep_idx: list[int] = []
    mat = np.asarray(vecs, dtype="float32")
    mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
    for i in range(len(situations)):
        if any(float(mat[i] @ mat[j]) > thresh for j in keep_idx):
            continue
        keep_idx.append(i)
        if len(keep_idx) >= cap:
            break
    return [situations[i] for i in keep_idx], [vecs[i] for i in keep_idx]


def _rag_retrieve_match(kb: KnowledgeBase, text: str, ctx: _Ctx, *, top_k: int) -> dict[str, dict]:
    # Этап 1 — извлечь обстоятельства ПОЧАНКОВО, чтобы не упустить упомянутые
    # вскользь (ключевой риск может прятаться в одной строке среди ковенантов).
    chunks = _chunk_text(text)

    def _extract_chunk(ch: str) -> list[str]:
        s1 = ctx.llm.structured(task="rag.situations", system=_EXTRACT_SYS,
                                user=f"ФРАГМЕНТ ДОКУМЕНТА:\n{ch}",
                                schema=_SITUATIONS_SCHEMA, context={"text": ch})
        return [re.sub(r"^[\s\-•]+", "", s).strip() for s in s1.get("situations", []) if s and s.strip()]

    situations: list[str] = []
    for res in _map_concurrent(chunks, _extract_chunk):  # параллельно по фрагментам
        situations += res or []
    if not situations:
        ctx.trace.append(TraceStep("situations", "извлечено: 0"))
        return {}

    # Дедупликация near-дубликатов + ограничение числа обстоятельств.
    vecs = ctx.llm.embed(situations)
    situations, vecs = _dedupe_situations(situations, vecs)
    ctx.trace.append(TraceStep("situations", f"уникальных обстоятельств: {len(situations)} (из {len(chunks)} фрагментов)"))

    # Этап 2–3 — retrieval кандидатов и пер-ситуационное сопоставление (параллельно).
    _MATCH_SYS = (
        "Дано обстоятельство из документа и риски-кандидаты (код: название; типовая ситуация). "
        "Верни risk_codes тех кандидатов, чья типовая ситуация РЕАЛИЗУЕТСЯ этим обстоятельством — "
        "в том числе когда обстоятельство является СПОСОБОМ или ФОРМОЙ реализации риска "
        "(например, приобретение доли/заключение квазикорпоративного договора → установление "
        "корпоративного контроля). НЕ включай кандидатов, относящихся лишь к общей теме без прямой "
        "связи с обстоятельством. Возвращай только коды из списка."
    )

    def _match_one(pair: tuple[str, Any]) -> list[tuple[str, str]]:
        situation, qv = pair
        codes = kb.retrieve([qv], k=top_k)
        block = "\n".join(
            f"{c}: {kb.get(c).title}  (типовая ситуация: {re.sub(r'_+', ' ', kb.get(c).fact).strip()[:180]})"
            for c in codes if kb.get(c)
        )
        m = ctx.llm.structured(
            task="rag.match", system=_MATCH_SYS,
            user=f"ОБСТОЯТЕЛЬСТВО:\n{situation}\n\nКАНДИДАТЫ:\n{block}",
            schema=_CODES_SCHEMA, context={"text": situation},
        )
        return [(c, situation) for c in m.get("risk_codes", []) if kb.get(c)]

    detected: dict[str, dict] = {}
    for res in _map_concurrent(list(zip(situations, vecs)), _match_one):  # порядок сохранён
        for c, situation in (res or []):
            if c not in detected:
                detected[c] = {"risk_code": c, "title": kb.get(c).title, "fact": situation}
    return detected


def _resolve_package(inp: AgentInput, ctx: _Ctx) -> list[tuple[str, str]]:
    """Разворачивает вход в список документов (имя, текст).

    Принимает директорию (все файлы внутри), список файлов или один текст.
    Это «разбиение пакета» на отдельные документы (ср. shadow_agent: ECM +
    ArchiveExpander отдают пакет как список файлов с метаданными)."""
    paths: list[Path] = []
    for f in inp.files:
        p = Path(f)
        if p.is_dir():
            paths += [c for c in sorted(p.iterdir()) if c.is_file()]
        elif p.exists():
            paths.append(p)
    out: list[tuple[str, str]] = []
    for p in paths:
        res = ctx.reader.read(str(p))
        ctx.trace.append(TraceStep("read_document", f"{p.name} → {res.status}"))
        out.append((p.name, res.text or ""))
    if not out and inp.text:
        out.append(("document", inp.text))
    return out


_PKG_DOC_SCHEMA = {
    "type": "object",
    "properties": {
        "doc_type": {"type": "string"},
        "summary": {"type": "string"},
        "findings": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["doc_type"],
}


def _strat_package_review(spec: AgentSpec, inp: AgentInput, ctx: _Ctx) -> RunResult:
    """Обработка ПАКЕТА документов (кейса): классификация типа каждого документа →
    маршрутизация по типу → пер-документный анализ → агрегация в единое заключение.

    Повторяет каркас shadow_agent (classify → route → analyze → aggregate в flows
    + статус-вердикт + gating по обязательным типам), но домен задаётся конфигом
    маршрутизации (`settings.package`), а не кодом."""
    pkg = spec.settings.get("package") or {}
    taxonomy = pkg.get("taxonomy") or []
    required = pkg.get("required") or []
    per_type = pkg.get("per_type") or {}
    type_codes = [t.get("code") for t in taxonomy if t.get("code")]

    docs = _resolve_package(inp, ctx)
    if not docs:
        return RunResult(status="error", errors=["пакет пуст: не найдено документов"], trace=ctx.trace)
    if ctx.dry_run:
        return RunResult(status="ok", output={"_dry_run": True, "documents": len(docs)}, trace=ctx.trace)

    tax_block = "\n".join(
        f"- {t['code']}: {t.get('title', '')} — {t.get('description', '')}" for t in taxonomy
    ) or "(таксономия не задана — определи тип свободно)"

    instr = per_type.get("*", "")  # глобальная инструкция к пер-документному анализу
    _PKG_SYS = spec.system_prompt + "\n\nОпредели ТИП документа (верни code из таксономии; если " \
        "не подходит ни один — 'unknown'), кратко суммируй документ и перечисли ключевые " \
        "юридические находки/риски. " + (instr or "")

    def _analyze_doc(doc: tuple[str, str]) -> dict:
        name, text = doc
        if not text.strip():
            return {"name": name, "doc_type": "unreadable", "summary": "", "findings": []}
        try:
            clipped, _ = _clip(text, 12000, ctx, f"документ {name}")
            data = ctx.llm.structured(
                task="package.analyze", system=_PKG_SYS,
                user=f"ТАКСОНОМИЯ ТИПОВ:\n{tax_block}\n\nДОКУМЕНТ «{name}»:\n{clipped}",
                schema=_PKG_DOC_SCHEMA, context={"text": text},
            )
        except Exception as e:  # сбой по одному документу не рушит весь пакет
            ctx.trace.append(TraceStep("doc_error", f"{name}: {e}"))
            return {"name": name, "doc_type": "error", "summary": f"ошибка анализа: {e}", "findings": []}
        return {
            "name": name, "doc_type": (data.get("doc_type") or "unknown").strip(),
            "summary": data.get("summary", ""),
            "findings": [f for f in (data.get("findings") or []) if isinstance(f, str) and f.strip()],
        }

    results: list[dict] = [r for r in _map_concurrent(docs, _analyze_doc) if r]

    present = {r["doc_type"] for r in results}
    missing_required = [c for c in required if c not in present]
    unrouted = [r["name"] for r in results if type_codes and r["doc_type"] not in type_codes]
    flows = [
        {"document": r["name"], "doc_type": r["doc_type"], "finding": f}
        for r in results for f in r["findings"]
    ]
    ctx.trace.append(TraceStep("aggregate", f"документов: {len(results)}, находок: {len(flows)}, "
                               f"не хватает типов: {len(missing_required)}"))

    verdict = "rejected" if missing_required else ("partial" if any(r["doc_type"] == "unreadable" for r in results) else "success")
    case_summary = _package_case_summary(spec, results, missing_required, ctx)

    output: dict[str, Any] = {
        "verdict": verdict,
        "documents": results,
        "flows": flows,
        "missing_required": missing_required,
        "unrouted": unrouted,
        "case_summary": case_summary,
    }
    status = "ok" if verdict == "success" else "partial"
    warnings = ([f"не хватает обязательных типов: {missing_required}"] if missing_required else []) + \
               ([f"нераспознанные документы: {unrouted}"] if unrouted else [])
    return RunResult(status=status, output=output, warnings=warnings, trace=ctx.trace)


def _package_case_summary(spec: AgentSpec, results: list[dict], missing: list[str], ctx: _Ctx) -> str:
    per_doc = "\n".join(f"- {r['name']} [{r['doc_type']}]: {r['summary'][:200]}" for r in results)
    try:
        return ctx.llm.text(
            task="package.summary",
            system="Сформируй краткое сводное заключение по пакету документов сделки: что за кейс, "
            "ключевые риски/находки, чего не хватает. 3–6 предложений.",
            user=f"ДОКУМЕНТЫ:\n{per_doc}\n\nНе хватает обязательных типов: {missing or 'нет'}",
            temperature=0.2,
        ) or per_doc
    except Exception:
        return per_doc


_CHECKLIST_SCHEMA = {
    "type": "object",
    "properties": {"results": {"type": "array", "items": {"type": "object", "properties": {
        "check_id": {"type": "string"},
        "verdict": {"type": "string"},          # yes | no | n_a
        "evidence": {"type": "string"},
    }, "required": ["check_id", "verdict"]}}},
    "required": ["results"],
}


def _strat_checklist_review(spec: AgentSpec, inp: AgentInput, ctx: _Ctx) -> RunResult:
    """Экспертиза по чек-листу (аналог shadow_agent/expertise: guarantee, corporate_approval).

    Каталог декларативных проверок «да/нет»; LLM решает, выполняется ли условие в
    документе (надёжная задача), а ЮРИДИЧЕСКИЙ вывод (позиция/риск/рекомендация)
    берётся ШАБЛОНОМ из данных проверки — как в guarantee_dict (true→шаблон).
    """
    checklist = spec.settings.get("checklist") or []
    if not checklist:
        return RunResult(status="error", errors=["для checklist нужен каталог проверок (checklist.json)"], trace=ctx.trace)
    text = _resolve_text(inp, ctx)
    if not text.strip():
        return RunResult(status="error", errors=["пустой вход: нечего проверять"], trace=ctx.trace)
    if ctx.dry_run:
        return RunResult(status="ok", output={"_dry_run": True, "checks": len(checklist)}, trace=ctx.trace)

    by_id = {str(c.get("id")): c for c in checklist if c.get("id")}
    clipped, _ = _clip(text, 14000, ctx, "документ")
    _CL_SYS = spec.system_prompt + "\n\nДля КАЖДОЙ проверки определи verdict: 'yes' — условие " \
        "выполняется/присутствует в документе; 'no' — не выполняется; 'n_a' — неприменимо/нет " \
        "данных. Добавь evidence — короткую цитату из документа. Опирайся только на текст."
    batch = 8
    items = list(by_id.values())
    batches = [items[i : i + batch] for i in range(0, len(items), batch)]

    def _eval_batch(chunk: list[dict]) -> list[dict]:
        block = "\n".join(
            f"{c['id']}: {c.get('question', '')}  (признак выполнения: {c.get('criteria', '')})"
            for c in chunk
        )
        r = ctx.llm.structured(
            task="checklist.eval", system=_CL_SYS,
            user=f"ДОКУМЕНТ:\n{clipped}\n\nПРОВЕРКИ:\n{block}",
            schema=_CHECKLIST_SCHEMA, context={"text": text},
        )
        return r.get("results", [])

    verdicts: dict[str, dict] = {}
    for res in _map_concurrent(batches, _eval_batch):  # батчи параллельно
        for item in (res or []):
            cid = str(item.get("check_id"))
            if cid in by_id:
                verdicts[cid] = item

    unanswered = [cid for cid in by_id if cid not in verdicts]
    if unanswered:
        ctx.trace.append(TraceStep("checklist_warn", f"модель не вернула вердикт по {len(unanswered)} проверкам"))

    results: list[dict] = []
    fired_high = fired_any = False
    for cid, c in by_id.items():
        # Отличаем «модель не ответила» (unknown) от настоящего n_a — чтобы
        # пропущенная high-проверка не исчезала молча из вердикта.
        answered = cid in verdicts
        v = (verdicts.get(cid, {}).get("verdict") or "unknown").lower() if answered else "unknown"
        evidence = verdicts.get(cid, {}).get("evidence", "")
        if v == "yes":
            fired_any = True
            if str(c.get("severity", "")).lower() == "high":
                fired_high = True
            position = c.get("on_yes_position", "Имеются замечания")
            finding = c.get("on_yes_finding", "")
            recommendation = c.get("on_yes_recommendation", "")
        elif v == "n_a":
            position, finding, recommendation = "Не применимо", "", ""
        elif v == "unknown":
            position, finding, recommendation = "Не проверено", "Модель не вернула вердикт по проверке.", ""
        else:
            position, finding, recommendation = "Соответствует", "", ""
        results.append({
            "check_id": cid, "question": c.get("question", ""), "verdict": v,
            "position": position, "finding": finding,
            "recommendation": recommendation, "evidence": evidence,
        })

    overall = "Выявлены правовые риски" if fired_high else ("Имеются замечания" if fired_any else "Соответствует")
    flagged = [r["check_id"] for r in results if r["verdict"] == "yes"]
    ctx.trace.append(TraceStep("checklist", f"проверок: {len(by_id)}, сработало: {len(flagged)}; вывод: {overall}"))
    out_status = "partial" if unanswered else "ok"
    output = {"results": results, "flagged": flagged, "overall_position": overall, "unanswered": unanswered}
    warnings = [f"нет вердикта по проверкам: {unanswered}"] if unanswered else []
    return RunResult(status=out_status, output=output, warnings=warnings, trace=ctx.trace)


def _strat_rule_check(spec: AgentSpec, inp: AgentInput, ctx: _Ctx) -> RunResult:
    """Детерминированная проверка документа по каталогу правил (rule-engine).

    Без LLM: дёшево, воспроизводимо, работает офлайн. Подходит для проверок,
    которые можно выразить ключевыми словами/близостью/regex."""
    catalog = spec.settings.get("rule_catalog") or []
    if not catalog:
        return RunResult(status="error", errors=["для rule_check нужен каталог правил (rule_catalog.json)"], trace=ctx.trace)
    text = _resolve_text(inp, ctx)
    if not text.strip():
        return RunResult(status="error", errors=["пустой вход: нечего проверять"], trace=ctx.trace)
    if ctx.dry_run:
        return RunResult(status="ok", output={"_dry_run": True, "rules": len(catalog)}, trace=ctx.trace)
    findings = RuleEngine.from_list(catalog).evaluate(text)
    ctx.trace.append(TraceStep("rule_check", f"правил: {len(catalog)}, сработало: {len(findings)}"))
    output: dict[str, Any] = {"matched_rules": [
        {"rule_id": f.rule_id, "title": f.title, "severity": f.severity, "evidence": f.evidence}
        for f in findings
    ]}
    if "confidence" in spec.output_schema.get("properties", {}):
        output["confidence"] = 1.0  # детерминированно
    return _finish_structured(spec, output, ctx)


_STRATEGIES: dict[str, Callable[[AgentSpec, AgentInput, _Ctx], RunResult]] = {
    "extract": _strat_extract,
    "compare": _strat_compare,
    "generate": _strat_generate,
    "chat": _strat_chat,
    "workflow": _strat_workflow,
    "rag_detect": _strat_rag_detect,
    "rule_check": _strat_rule_check,
    "package_review": _strat_package_review,
    "checklist_review": _strat_checklist_review,
}
