"""Библиотека среды выполнения, импортируемая сгенерированными агентами.

Упакованный агент — это данные (промпт + схема + конфигурация). Он загружается
через :func:`AgentSpec.load` и запускается через :func:`run_agent`, который
выполняет разрешение входных данных, валидацию по схеме, оценку уверенности,
ведение трассы выполнения и холостой прогон.
"""

from __future__ import annotations

from .executor import AgentInput, run_agent
from .spec import AgentSpec, RunResult, TraceStep

__all__ = ["AgentSpec", "AgentInput", "RunResult", "TraceStep", "run_agent"]
