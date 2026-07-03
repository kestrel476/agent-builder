"""Адаптеры над двумя переиспользуемыми сервисами (LLM + извлечение текста из документов).

Остальная часть Forge зависит только от небольших интерфейсов, определённых здесь,
и никогда напрямую от ``devassist``/``extractors``. Это позволяет легко заменять
заимствованный код и даёт возможность всему конвейеру работать офлайн через
детерминированный офлайн-движок (заглушку).
"""

from __future__ import annotations

from .documents import DocumentReader, ReadResult
from .llm import ForgeLLM, LLMUnavailable, build_llm

__all__ = [
    "ForgeLLM",
    "LLMUnavailable",
    "build_llm",
    "DocumentReader",
    "ReadResult",
]
