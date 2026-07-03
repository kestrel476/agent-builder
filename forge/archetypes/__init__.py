"""Реестр архетипов. Импорт этого пакета регистрирует встроенные архетипы."""

from __future__ import annotations

from . import builtin  # noqa: F401  (побочный эффект: регистрирует встроенные архетипы)
from .base import AgentArchetype, all_archetypes, get, keys, register

__all__ = ["AgentArchetype", "all_archetypes", "get", "keys", "register"]
