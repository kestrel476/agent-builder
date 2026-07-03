"""Декларативный rule-engine: правила-как-данные с подключаемыми матчерами.

Заимствовано (как ПАТТЕРН, не код) из rule-engine lex_copilot: правила
описываются данными (JSON), а не кодом; один интерфейс матчера обслуживает и
текстовые проверки, и (опционально) LLM. Это детерминированный, дешёвый и
работающий офлайн контур — антидот к вариативности LLM-детекции: чем можно
проверить правилом, проверяем правилом.

Поддерживаемые типы матча (`match`):
* ``any``   — встретился ЛЮБОЙ из ``terms``;
* ``all``   — встретились ВСЕ ``terms`` (в любом месте);
* ``near``  — все ``terms`` встретились в окне ``window`` символов (proximity);
* ``regex`` — сработал ``pattern``.

Поле ``scope`` (опц.) ограничивает поиск секцией: текст после заголовка,
содержащего ``scope`` (до следующего заголовка). ``negate`` инвертирует.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RuleSpec:
    id: str
    title: str = ""
    severity: str = "medium"           # low | medium | high
    match: str = "any"                 # any | all | near | regex
    terms: list[str] = field(default_factory=list)
    pattern: str = ""                  # для match=regex
    window: int = 200                  # для match=near (символы)
    scope: str = ""                    # ограничить поиск секцией с этим заголовком
    negate: bool = False               # сработать, когда условие НЕ выполняется

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RuleSpec":
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**known)


@dataclass
class RuleFinding:
    rule_id: str
    title: str
    severity: str
    evidence: str = ""


class RuleEngine:
    """Оценивает каталог правил по тексту документа и возвращает сработавшие."""

    def __init__(self, rules: list[RuleSpec]) -> None:
        self.rules = rules

    @classmethod
    def from_list(cls, items: list[dict[str, Any]]) -> "RuleEngine":
        return cls([RuleSpec.from_dict(it) for it in items if it.get("id")])

    def evaluate(self, text: str) -> list[RuleFinding]:
        findings: list[RuleFinding] = []
        for rule in self.rules:
            scoped = _scope_text(text, rule.scope) if rule.scope else text
            hit, pos = _match(rule, scoped)
            if rule.negate:
                hit = not hit
            if hit:
                findings.append(RuleFinding(
                    rule_id=rule.id, title=rule.title, severity=rule.severity,
                    evidence=_evidence(scoped, pos) if pos is not None else "",
                ))
        return findings


# --------------------------------------------------------------------------- #
# Матчеры
# --------------------------------------------------------------------------- #
def _match(rule: RuleSpec, text: str) -> tuple[bool, int | None]:
    low = text.lower()
    if rule.match == "regex":
        m = re.search(rule.pattern, text, re.I | re.S) if rule.pattern else None
        return (bool(m), m.start() if m else None)

    terms = [t.lower() for t in rule.terms if t.strip()]
    if not terms:
        return (False, None)
    positions = {t: low.find(t) for t in terms}

    if rule.match == "any":
        present = [(t, p) for t, p in positions.items() if p >= 0]
        if not present:
            return (False, None)
        return (True, min(p for _, p in present))

    if rule.match == "all":
        if any(p < 0 for p in positions.values()):
            return (False, None)
        return (True, min(positions.values()))

    if rule.match == "near":
        if any(p < 0 for p in positions.values()):
            return (False, None)
        # все термины должны уместиться в окно window символов
        return _near(low, terms, rule.window)

    return (False, None)


def _near(low: str, terms: list[str], window: int) -> tuple[bool, int | None]:
    """Все термины присутствуют в окне ``window`` символов хотя бы раз."""
    occ = {t: [m.start() for m in re.finditer(re.escape(t), low)] for t in terms}
    anchor = terms[0]
    for a in occ[anchor]:
        ok = True
        lo, hi = a, a + len(anchor)
        for t in terms[1:]:
            near_pos = [p for p in occ[t] if abs(p - a) <= window]
            if not near_pos:
                ok = False
                break
            lo = min(lo, min(near_pos))
            hi = max(hi, max(p + len(t) for p in near_pos))
        if ok:
            return (True, lo)
    return (False, None)


# --------------------------------------------------------------------------- #
# Вспомогательное
# --------------------------------------------------------------------------- #
def _scope_text(text: str, scope: str) -> str:
    """Текст секции, чей заголовок содержит ``scope`` (до следующего заголовка)."""
    lines = text.splitlines()
    out: list[str] = []
    capturing = False
    for ln in lines:
        is_heading = bool(ln.strip()) and (ln.strip().endswith(":") or ln.isupper()
                                           or re.match(r"^\s*(статья|раздел|приложение|п\.|§)\b", ln, re.I))
        if is_heading:
            if capturing:
                break
            if scope.lower() in ln.lower():
                capturing = True
                continue
        if capturing:
            out.append(ln)
    return "\n".join(out) if out else text  # если секция не найдена — весь текст


def _evidence(text: str, pos: int, span: int = 90) -> str:
    start = max(0, pos - span // 3)
    snippet = text[start: pos + span].strip()
    return " ".join(snippet.split())
