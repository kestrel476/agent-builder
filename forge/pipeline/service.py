"""Фасад :class:`Forge` — единый объект, которым управляет CLI.

Содержит общий «мозг» (LLM), читатель документов и движок уточнений и
предоставляет по одному методу на каждый этап конвейера плюс цикл
тестирования/доработки. Делает CLI тонким и позволяет описывать/тестировать весь
конвейер в несколько строк.
"""

from __future__ import annotations


from ..clarify import ClarifyEngine
from ..integrations import DocumentReader, ForgeLLM, build_llm
from ..model import AgentBlueprint
from ..store import Workspace
from . import contract as _contract
from . import intake as _intake
from . import packaging as _packaging
from . import synthesis as _synthesis
from . import testing as _testing
from .testing import TestReport

__all__ = ["Forge", "TestReport"]


class Forge:
    def __init__(self, llm: ForgeLLM | None = None, *, offline: bool = False,
                 model: str | None = None) -> None:
        self.llm = llm or build_llm(offline=offline, model=model)
        self.reader = DocumentReader()
        self.clarify = ClarifyEngine(self.llm)

    # -- этап 0 ----------------------------------------------------------- #
    def intake(self, bp: AgentBlueprint):
        return _intake.run_intake(bp, self.llm)

    def detect_archetype(self, raw: str):
        return _intake.detect_archetype(raw, self.llm)

    # -- уточнение -------------------------------------------------------- #
    def analyze(self, bp: AgentBlueprint, *, use_llm: bool = True):
        return self.clarify.analyze(bp, use_llm=use_llm)

    def answer(self, bp: AgentBlueprint, qid: str, text: str) -> None:
        self.clarify.answer(bp, qid, text)

    def assume_open(self, bp: AgentBlueprint, *, only_blocking: bool = False):
        return self.clarify.assume_open(bp, only_blocking=only_blocking)

    # -- этап 1 ----------------------------------------------------------- #
    def build_contract(self, bp: AgentBlueprint) -> None:
        _contract.build_contract(bp, self.llm)

    # -- этап 2 ----------------------------------------------------------- #
    def synthesize(self, bp: AgentBlueprint, ws: Workspace):
        return _synthesis.synthesize(bp, ws, self.llm)

    # -- этап 3 ----------------------------------------------------------- #
    def test(self, bp: AgentBlueprint, ws: Workspace, *, only: str | None = None) -> TestReport:
        return _testing.run_tests(bp, ws, self.llm, self.reader, only=only)

    def diagnose(self, bp: AgentBlueprint, report: TestReport):
        return _testing.diagnose(bp, report, self.llm)

    def refine(self, bp: AgentBlueprint, diagnoses) -> str:
        return _testing.refine(bp, diagnoses, self.llm)

    def test_and_refine(self, bp: AgentBlueprint, ws: Workspace, *, max_iters: int = 3):
        """Полный цикл: тест → диагностика → доработка → пересинтез, пока не станет
        зелёным или прогресс не прекратится. Возвращает (итерация, отчёт, диагнозы)
        для наблюдаемости."""
        history = []
        prev_passed = -1
        for i in range(1, max_iters + 1):
            report = self.test(bp, ws)
            history.append((i, report, []))
            if report.green or report.passed <= prev_passed:
                if report.green:
                    break
                # нет улучшения → прекращаем доработку, возвращаем что есть
                diagnoses = self.diagnose(bp, report)
                history[-1] = (i, report, diagnoses)
                break
            prev_passed = report.passed
            diagnoses = self.diagnose(bp, report)
            history[-1] = (i, report, diagnoses)
            self.refine(bp, diagnoses)
            self.synthesize(bp, ws)  # пересборка с дополнением
        return history

    # -- этап 4 ----------------------------------------------------------- #
    def package(self, bp: AgentBlueprint, ws: Workspace, last_report: TestReport | None = None):
        return _packaging.package(bp, ws, last_report)

    # -- запуск сгенерированного агента по требованию ---------------------- #
    def run_agent_bundle(self, ws: Workspace, value, *, dry_run: bool = False):
        from ..runtime import AgentSpec, run_agent
        spec = AgentSpec.load(ws.agent_dir)
        return run_agent(spec, value, llm=self.llm, reader=self.reader, dry_run=dry_run)
