"""Движок уточнения: суть ценности Forge.

Он (1) запускает критиков, чтобы найти слабые места, (2) преобразует их и
промпты архетипа/LLM в приоритизированные вопросы, (3) фиксирует ответы и
(4) — при неинтерактивном запуске — превращает оставшиеся без ответа
*блокирующие* вопросы в явные, помеченные риском допущения, вместо того чтобы
угадывать молча.
"""

from __future__ import annotations

from ..archetypes import get as get_archetype
from ..integrations import ForgeLLM
from ..model import (
    AgentBlueprint,
    Assumption,
    Issue,
    IssueStatus,
    Question,
    Severity,
    Stage,
)
from .critics import DETERMINISTIC_CRITICS, llm_questions

_SEV_RANK = {Severity.BLOCKER: 0, Severity.MAJOR: 1, Severity.MINOR: 2, Severity.INFO: 3}


class ClarifyEngine:
    def __init__(self, llm: ForgeLLM) -> None:
        self.llm = llm

    # ------------------------------------------------------------------ #
    def analyze(self, bp: AgentBlueprint, *, use_llm: bool = True) -> list[Issue]:
        """Запустить всех критиков, влить замечания + вопросы в чертёж."""
        new_issues: list[Issue] = []
        for critic in DETERMINISTIC_CRITICS:
            for issue in critic.run(bp):
                stored = bp.add_issue(issue)
                if stored is issue:
                    new_issues.append(stored)

        # Засеять базовые вопросы архетипа ровно один раз.
        if not any(q.id.startswith("Q-ARCH") for q in bp.questions):
            for q in get_archetype(bp.archetype).baseline_questions(bp):
                self._add_question(bp, q)

        # Превратить новые блокирующие/важные замечания в вопросы.
        for issue in bp.open_issues:
            if issue.severity in (Severity.BLOCKER, Severity.MAJOR):
                self._ensure_question_for_issue(bp, issue)

        # Запросить у «мозга» несколько более глубоких вопросов (пропускается
        # офлайн ради детерминизма, если явно не включено).
        if use_llm and not self.llm.offline:
            for q in llm_questions(bp, self.llm):
                self._add_question(bp, Question(
                    id=bp.next_id("Q"), text=q["text"], why=q.get("why", ""),
                    category=q.get("category", "general"), blocking=bool(q.get("blocking", False))))

        if bp.stage in (Stage.DRAFT, Stage.INTAKE) and bp.open_questions:
            bp.stage = Stage.CLARIFYING
        bp.record("analyze", f"{len(new_issues)} новых замечаний; {len(bp.open_questions)} открытых вопросов")
        return new_issues

    # ------------------------------------------------------------------ #
    def pending_questions(self, bp: AgentBlueprint) -> list[Question]:
        """Открытые вопросы, самые важные первыми."""
        def rank(q: Question) -> tuple:
            issue = self._issue(bp, q.issue_id)
            sev = _SEV_RANK.get(issue.severity, 2) if issue else (0 if q.blocking else 2)
            return (0 if q.blocking else 1, sev, q.id)
        return sorted(bp.open_questions, key=rank)

    def answer(self, bp: AgentBlueprint, qid: str, text: str) -> None:
        for q in bp.questions:
            if q.id == qid:
                q.answer = text
                if q.issue_id:
                    bp.resolve_issue(q.issue_id, IssueStatus.RESOLVED)
                bp.record("answer", f"{qid}: {text[:80]}")
                return
        raise KeyError(f"Нет вопроса {qid}")

    def assume_open(self, bp: AgentBlueprint, *, only_blocking: bool = False) -> list[Assumption]:
        """Превратить оставшиеся без ответа вопросы в явные допущения (для --auto)."""
        made: list[Assumption] = []
        for q in self.pending_questions(bp):
            if only_blocking and not q.blocking:
                continue
            issue = self._issue(bp, q.issue_id)
            default = (issue.suggestion if issue and issue.suggestion else "Принять консервативное значение по умолчанию.")
            asm = Assumption(
                id=bp.next_id("ASM"),
                statement=f"Для \"{q.text}\" — допущение: {default}",
                because="Ответ не был предоставлен; зафиксировано как явное допущение вместо молчаливого угадывания.",
                risk=Severity.MAJOR if q.blocking else Severity.MINOR,
                issue_id=q.issue_id,
            )
            bp.assumptions.append(asm)
            q.answer = f"[допущение] {default}"
            if q.issue_id:
                bp.resolve_issue(q.issue_id, IssueStatus.ASSUMED)
            made.append(asm)
        if made:
            bp.record("assume", f"зафиксировано {len(made)} допущений")
        return made

    # ------------------------------------------------------------------ #
    def _ensure_question_for_issue(self, bp: AgentBlueprint, issue: Issue) -> None:
        if any(q.issue_id == issue.id for q in bp.questions):
            return
        text = issue.message
        if issue.suggestion:
            text = f"{issue.message} (предложение: {issue.suggestion})"
        self._add_question(bp, Question(
            id=bp.next_id("Q"), text=text, why=issue.rationale,
            category=issue.kind.value, blocking=issue.severity == Severity.BLOCKER,
            issue_id=issue.id))

    @staticmethod
    def _add_question(bp: AgentBlueprint, q: Question) -> None:
        if any(existing.text == q.text for existing in bp.questions):
            return
        bp.questions.append(q)

    @staticmethod
    def _issue(bp: AgentBlueprint, issue_id: str | None) -> Issue | None:
        if not issue_id:
            return None
        return next((i for i in bp.issues if i.id == issue_id), None)
