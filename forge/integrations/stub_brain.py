"""Детерминированный офлайн-«движок» (заглушка).

Это **не** попытка имитировать LLM. Это воспроизводимый запасной вариант, чтобы
Forge — и каждый генерируемый им агент — продолжал работать без сети и без
учётных данных: полезно для разработки, CI, демонстраций и набора тестов. При
наличии реальных учётных данных GigaChat этот модуль никогда не используется.

Он выполняет две роли:

* **Рассуждения на этапе Forge** (структурирование инструкций, предложение
  вопросов, диагностика провалов тестов) — эвристические, но правдоподобные.
* **Извлечение на этапе работы агента** — по JSON-схеме и тексту документа он
  заполняет схему с помощью лёгких regex (даты, деньги, стороны, e-mail, …),
  чтобы сгенерированный агент извлечения выдавал правдоподобный вывод офлайн.

Всё опирается на строку ``task`` и структурированный словарь ``context``,
поэтому результаты стабильны между запусками.
"""

from __future__ import annotations

import re
from typing import Any, Callable

_DATE_RE = re.compile(
    r"\b(\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{4}-\d{2}-\d{2}|"
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b"
)
_MONEY_RE = re.compile(r"(?:USD|EUR|GBP|RUB|\$|€|£)\s?\d[\d ,.]*\d|\b\d[\d ,.]*\d\s?(?:USD|EUR|GBP|RUB|dollars|euros)\b", re.I)
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PARTY_RE = re.compile(
    r'\b(?:between|by and between|This Agreement.*?between)\s+(.+?)\s+(?:and|&)\s+(.+?)[\s,.(]', re.I | re.S
)
# Юридические лица по их корпоративному суффиксу — устойчиво для текста договоров.
_ENTITY_RE = re.compile(
    r'\b((?:[A-Z][\w&.\-]*\s+){0,4}'
    r'(?:Incorporated|Inc|LLC|L\.L\.C|Limited|Ltd|Corporation|Corp|Company|Co|GmbH|LLP|LP|PLC|S\.A|N\.V)\.?)',
)
_TERM_RE = re.compile(r"\b(\d+)\s*(day|week|month|year)s?\b", re.I)

# Ключевые слова, указывающие на уровень строки инструкции.
_RULE_HINTS = ("must", "shall", "should", "require", "only if", "unless", "if ", "обязан", "должен")
_CONSTRAINT_HINTS = ("never", "do not", "don't", "no ", "prohibit", "не ", "запрещ", "confidential")
_AMBIGUITY_HINTS = ("etc", "and so on", "as appropriate", "reasonable", "relevant", "where applicable",
                    "if needed", "and/or", "various", "some", "и т.д", "при необходимости")


class StubBrain:
    def __init__(self) -> None:
        self._structured: dict[str, Callable[[dict, dict, str], dict]] = {
            "intake.structure": self._intake_structure,
            "intake.detect_archetype": self._detect_archetype,
            "clarify.questions": self._clarify_questions,
            "contract.output_schema": self._output_schema,
            "contract.categories": self._categories,
            "rag.situations": self._rag_situations,
            "rag.match": self._rag_match,
            "package.analyze": self._package_analyze,
            "checklist.eval": self._checklist_eval,
            "test.diagnose": self._diagnose,
            "refine.propose": self._refine,
        }
        self._text: dict[str, Callable[[dict, str], str]] = {
            "synthesis.system_prompt": self._system_prompt,
        }

    # ------------------------------------------------------------------ #
    def structured(self, *, task: str, context: dict, schema: dict, user: str) -> dict:
        handler = self._structured.get(task)
        if handler is not None:
            return handler(context, schema, user)
        # Общий случай: извлечение на этапе работы агента или любая неизвестная структурированная задача.
        return generic_fill(schema, context.get("text", "") or user, context)

    def text(self, *, task: str, context: dict, user: str) -> str:
        handler = self._text.get(task)
        if handler is not None:
            return handler(context, user)
        return ""

    # ------------------------- задачи этапа forge ------------------------- #
    def _intake_structure(self, context: dict, schema: dict, user: str) -> dict:
        raw = context.get("raw", "") or user
        lines = [ln.strip(" -*•\t") for ln in re.split(r"[\n;]", raw) if ln.strip()]
        goal, rules, constraints, inputs, outputs, ambiguities, missing = "", [], [], [], [], [], []
        for ln in lines:
            low = ln.lower()
            if not goal and any(w in low for w in ("goal", "purpose", "need", "want", "build", "цель", "извлеч", "extract", "classify", "review", "check")):
                goal = ln
            if any(h in low for h in _RULE_HINTS):
                rules.append(ln)
            if any(h in low for h in _CONSTRAINT_HINTS):
                constraints.append(ln)
            if any(w in low for w in ("input", "receive", "given", "document", "file", "вход", "json")):
                inputs.append(ln)
            if any(w in low for w in ("output", "return", "produce", "result", "выход", "результат")):
                outputs.append(ln)
            if any(h in low for h in _AMBIGUITY_HINTS):
                ambiguities.append(ln)
        if not goal and lines:
            goal = lines[0]
        # Всё, что упомянуто, но нигде не определено, попадает в "missing".
        if "format" not in raw.lower():
            missing.append("Формат вывода / схема не указаны.")
        if not any("error" in ln.lower() or "fail" in ln.lower() for ln in lines):
            missing.append("Обработка ошибок / низкой уверенности не указана.")
        return {
            "goal": goal,
            "scope_in": inputs[:6],
            "scope_out": [],
            "inputs": inputs[:6],
            "outputs": outputs[:6],
            "business_rules": rules[:12],
            "constraints": constraints[:8],
            "definitions": [],
            "ambiguities": ambiguities[:8],
            "missing_data": missing,
            "interpretation_risks": ambiguities[:4],
        }

    def _detect_archetype(self, context: dict, schema: dict, user: str) -> dict:
        text = (context.get("raw", "") or user).lower()
        table = [
            ("catalog_risk_detection", ("каталог", "catalog", "перечень рисков", "база рисков", "из списка рисков", "risks.json", "по каталогу")),
            ("json_extraction", ("extract", "field", "json", "извлеч", "parse", "pull out")),
            ("risk_check", ("risk", "compliance", "violat", "flag", "check condition", "риск", "проверк")),
            ("classification", ("classif", "categor", "type of document", "label", "классиф", "тип документ")),
            ("clause_comparison", ("compare", "clause", "redline", "diff", "match provision", "сопостав", "редакц")),
            ("doc_generation", ("generate", "draft", "produce a document", "template", "сгенерир", "состав")),
            ("chat_assistant", ("chat", "assistant", "answer question", "q&a", "ассистент", "чат")),
            ("workflow", ("pipeline", "workflow", "multi-step", "case", "orchestrat", "процесс", "кейс")),
        ]
        scores = {key: sum(text.count(w) for w in words) for key, words in table}
        best = max(scores, key=lambda k: scores[k])
        if scores[best] == 0:
            best = "json_extraction"
        return {
            "archetype": best,
            "confidence": min(1.0, 0.4 + 0.15 * scores[best]),
            "rationale": f"совпадение по ключевым словам (офлайн-эвристика): score={scores[best]}",
        }

    def _clarify_questions(self, context: dict, schema: dict, user: str) -> dict:
        # Превращаем каждую уже найденную неоднозначность/пропуск в вопрос.
        qs = []
        for amb in context.get("ambiguities", [])[:5]:
            qs.append({
                "text": f"Фраза \"{_clip(amb)}\" неоднозначна. Каков точный, проверяемый смысл?",
                "why": "Расплывчатые формулировки приводят к невоспроизводимому поведению агента.",
                "category": "ambiguity",
                "blocking": True,
            })
        for miss in context.get("missing", [])[:5]:
            qs.append({
                "text": f"{miss} Пожалуйста, уточните.",
                "why": "Необходимо для детерминированной фиксации поведения агента.",
                "category": "missing",
                "blocking": True,
            })
        return {"questions": qs}

    def _output_schema(self, context: dict, schema: dict, user: str) -> dict:
        # Выводим поля из любых упоминаний имён, похожих на поля; в крайнем случае
        # используем общую форму извлечения для юридических документов.
        text = context.get("raw", "") or user
        found = sorted(set(re.findall(r"\b([a-z][a-z_]{2,30})\s*(?:=|:|\bfield\b)", text.lower())))
        if not found:
            found = ["parties", "effective_date", "term", "governing_law", "total_amount"]
        fields = [
            {"name": f, "type": _guess_field_type(f), "required": i < 2,
             "description": f"Извлечённое поле: {f.replace('_', ' ')}."}
            for i, f in enumerate(found[:15])
        ]
        return {"fields": fields}

    def _categories(self, context: dict, schema: dict, user: str) -> dict:
        # Берём перечисление после "тип:"/"категории:" и режем по запятым/«или».
        text = context.get("raw", "") or user
        m = re.search(r"(?:тип[уые]?|категори[июяй]+|на категории|types?|categories)\s*:?\s*(.+?)(?:\.|\n|$)",
                      text, re.I | re.S)
        cats: list[str] = []
        if m:
            parts = re.split(r",|;|/|\bили\b|\band\b|\bor\b", m.group(1))
            cats = [p.strip(" .»«\"'()").lower() for p in parts if p.strip()]
            cats = [c for c in cats if 1 < len(c) < 40][:12]
        return {"categories": cats}

    def _rag_situations(self, context: dict, schema: dict, user: str) -> dict:
        # Офлайн: «обстоятельства» — это содержательные предложения фрагмента.
        text = context.get("text", "") or user
        sents = [s.strip() for s in re.split(r"[.\n;]", text) if len(s.strip()) > 25]
        return {"situations": sents[:15]}

    def _rag_match(self, context: dict, schema: dict, user: str) -> dict:
        # Офлайн: сопоставление по лексическому пересечению обстоятельства и кандидата.
        sit = context.get("text", "")
        sit_words = set(re.findall(r"\w{4,}", sit.lower()))
        out: list[str] = []
        for line in user.splitlines():
            m = re.match(r"\s*([A-Za-zА-Яа-я0-9][\w-]*):\s*(.+)", line)
            if not m:
                continue
            code, rest = m.group(1), m.group(2)
            cand_words = set(re.findall(r"\w{4,}", rest.lower()))
            if sit_words and len(sit_words & cand_words) / len(sit_words) > 0.3:
                out.append(code)
        return {"risk_codes": out[:3]}

    def _package_analyze(self, context: dict, schema: dict, user: str) -> dict:
        # Офлайн: тип — по лексическому совпадению с таксономией из user; summary —
        # первое предложение; findings — предложения с «рисковыми» словами.
        text = context.get("text", "") or user
        words = set(re.findall(r"\w{4,}", text.lower()))
        best, best_score = "unknown", 0
        for m in re.finditer(r"^-\s*([\w-]+):\s*(.+)$", user, re.M):
            code, desc = m.group(1), m.group(2)
            score = len(words & set(re.findall(r"\w{4,}", desc.lower())))
            if score > best_score:
                best, best_score = code, score
        sents = [s.strip() for s in re.split(r"[.\n]", text) if len(s.strip()) > 25]
        findings = [s for s in sents if any(h in s.lower() for h in _RULE_HINTS + _CONSTRAINT_HINTS)][:5]
        return {"doc_type": best, "summary": sents[0][:200] if sents else "", "findings": findings}

    def _checklist_eval(self, context: dict, schema: dict, user: str) -> dict:
        # Офлайн: verdict='yes', если слова «признака выполнения» проверки
        # встречаются в тексте документа; иначе 'no'.
        text = context.get("text", "")
        doc_part = user.split("ПРОВЕРКИ:", 1)[0]
        doc_words = set(re.findall(r"\w{4,}", (text or doc_part).lower()))
        block = user.split("ПРОВЕРКИ:", 1)[-1]
        out = []
        for line in block.splitlines():
            m = re.match(r"\s*([\w.\-]+):\s*(.+)", line)
            if not m:
                continue
            cid, rest = m.group(1), m.group(2)
            cm = re.search(r"признак выполнения:\s*(.+?)\)?\s*$", rest)
            criteria = cm.group(1) if cm else rest
            crit_words = set(re.findall(r"\w{4,}", criteria.lower()))
            overlap = len(doc_words & crit_words)
            verdict = "yes" if crit_words and overlap / max(1, len(crit_words)) > 0.4 else "no"
            out.append({"check_id": cid, "verdict": verdict, "evidence": ""})
        return {"results": out}

    def _diagnose(self, context: dict, schema: dict, user: str) -> dict:
        diffs = context.get("diffs", [])
        first = diffs[0] if diffs else {}
        field = first.get("field", "?")
        return {
            "summary": f"{len(diffs)} поле(й) отличается(ются) от ожидаемого.",
            "root_cause": (
                f"Поле '{field}': значение агента не совпадает с ожидаемым. "
                "Вероятно, промпт извлечения недостаточно конкретен для этого поля, "
                "либо критерий успеха для поля неоднозначен."
            ),
            "fix_suggestion": (
                f"Уточните инструкцию для '{field}' (дайте явное определение и "
                "пример) или ослабьте проверку, если само ожидаемое значение спорно."
            ),
            "blueprint_patch": {"prompt_addendum": f"Будьте точны в отношении поля '{field}'; копируйте его дословно из источника."},
        }

    def _refine(self, context: dict, schema: dict, user: str) -> dict:
        addenda = []
        for d in context.get("diagnoses", []):
            patch = d.get("blueprint_patch") or {}
            if patch.get("prompt_addendum"):
                addenda.append(patch["prompt_addendum"])
        return {
            "prompt_addendum": " ".join(addenda),
            "notes": "Офлайн-доработка: добавлены указания по точности полей из провалившихся случаев.",
        }

    def _system_prompt(self, context: dict, user: str) -> str:
        goal = context.get("goal", "Обработать юридический документ.")
        rules = context.get("rules", [])
        rules_block = "\n".join(f"- {r}" for r in rules) if rules else "- Строго следуйте схеме вывода."
        return (
            f"Вы — специализированный агент юридической автоматизации.\nЦель: {goal}\n\n"
            f"Правила:\n{rules_block}\n\n"
            "Работайте только с предоставленным текстом документа. Если обязательное "
            "значение отсутствует, установите его в null и понизьте свою уверенность. "
            "Никогда не выдумывайте факты."
        )


def _guess_field_type(name: str) -> str:
    """Эвристически выбрать JSON-тип поля по его имени (для офлайн-вывода схемы)."""
    n = name.lower()
    if any(k in n for k in ("parties", "counterpart", "list", "items", "findings", "tags")):
        return "array"
    if n.startswith(("is_", "has_")) or any(k in n for k in ("flag", "present", "required_", "boolean")):
        return "boolean"
    if any(k in n for k in ("count", "quantity", "number_of", "qty")):
        return "integer"
    return "string"


# --------------------------------------------------------------------------- #
# Универсальный заполнитель схемы (извлечение на этапе работы агента, офлайн)
# --------------------------------------------------------------------------- #
_FIELD_EXTRACTORS: dict[str, Callable[[str], Any]] = {}


def _reg(*names: str):
    def deco(fn):
        for n in names:
            _FIELD_EXTRACTORS[n] = fn
        return fn
    return deco


@_reg("effective_date", "date", "start_date", "signing_date", "execution_date")
def _x_date(text: str):
    m = _DATE_RE.search(text)
    return m.group(0) if m else None


@_reg("total_amount", "amount", "price", "value", "consideration", "fee")
def _x_money(text: str):
    m = _MONEY_RE.search(text)
    return m.group(0).strip() if m else None


@_reg("email", "contact_email", "notice_email")
def _x_email(text: str):
    m = _EMAIL_RE.search(text)
    return m.group(0) if m else None


@_reg("term", "duration")
def _x_term(text: str):
    m = _TERM_RE.search(text)
    return m.group(0) if m else None


@_reg("parties", "party", "counterparties")
def _x_parties(text: str):
    # Предпочитаем именованные юридические лица (сохраняем корпоративный суффикс).
    entities = []
    for m in _ENTITY_RE.finditer(text):
        name = " ".join(m.group(1).split()).strip(" .")
        if name not in entities:
            entities.append(name)
        if len(entities) == 2:
            break
    if entities:
        return entities
    m = _PARTY_RE.search(text)
    if m:
        return [m.group(1).strip(), m.group(2).strip()]
    return None


@_reg("governing_law", "jurisdiction")
def _x_law(text: str):
    m = re.search(r"governed by the laws of ([A-Z][\w ]+)", text)
    return m.group(1).strip() if m else None


def generic_fill(schema: dict, text: str, context: dict) -> dict:
    """Детерминированно заполняет объект JSON-схемы из свободного текста."""
    out: dict[str, Any] = {}
    props = schema.get("properties", {})
    for name, spec in props.items():
        if name == "confidence":
            continue  # вычисляется ниже из покрытия, никогда не заполняется regex
        out[name] = _fill_value(name, spec, text, context)
    # Соглашение: уверенность отражает, какую часть схемы мы фактически заполнили.
    if "confidence" in props:
        present = sum(1 for v in out.values() if v not in (None, "", [], {}))
        denom = max(1, len(out))
        out["confidence"] = round(present / denom, 2)
    return out


def _fill_value(name: str, spec: dict, text: str, context: dict) -> Any:
    typ = spec.get("type", "string")
    if name in context and name != "text":
        return context[name]
    extractor = _FIELD_EXTRACTORS.get(name.lower())
    if extractor is not None:
        val = extractor(text)
        if val is not None:
            return val
    if typ == "string":
        return None
    if typ in ("number", "integer"):
        m = re.search(r"\b\d+(?:\.\d+)?\b", text)
        if m:
            return float(m.group(0)) if typ == "number" else int(float(m.group(0)))
        return 0
    if typ == "boolean":
        return False
    if typ == "array":
        return []
    if typ == "object":
        return generic_fill(spec, text, context)
    return None


def _clip(s: str, n: int = 60) -> str:
    s = s.strip()
    return s if len(s) <= n else s[: n - 1] + "…"
