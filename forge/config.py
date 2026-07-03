"""Конфигурация уровня Forge.

Отличается от конфигурации учётных данных GigaChat (она находится в переиспользуемом
пакете ``devassist`` и читается LLM-бэкендом). Здесь хранится только то, как ведёт
себя сам Forge: где находятся рабочие папки, какой мозг использовать, переключатели UX.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ENV_OFFLINE = "FORGE_OFFLINE"
ENV_HOME = "FORGE_HOME"


@dataclass
class ForgeConfig:
    home: Path           # корневая папка со всеми рабочими папками агентов (.forge)
    offline: bool        # принудительно использовать детерминированный заглушечный мозг
    model: str | None    # переопределение модели GigaChat
    assume_yes: bool     # автопринятие запросов (записывает допущения вместо вопросов)
    no_color: bool

    @classmethod
    def load(
        cls,
        *,
        home: str | os.PathLike | None = None,
        offline: bool | None = None,
        model: str | None = None,
        assume_yes: bool = False,
        no_color: bool = False,
    ) -> "ForgeConfig":
        root = Path(home or os.environ.get(ENV_HOME) or (Path.cwd() / ".forge")).resolve()
        off = offline if offline is not None else _env_bool(ENV_OFFLINE, default=False)
        return cls(home=root, offline=off, model=model, assume_yes=assume_yes, no_color=no_color)

    def ensure_home(self) -> Path:
        self.home.mkdir(parents=True, exist_ok=True)
        return self.home


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")
