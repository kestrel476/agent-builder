"""Поиск и импорт двух переиспользуемых соседних сервисов.

Мы намеренно *не* включаем копии ``devassist`` и ``extractors`` в проект.
Вместо этого мы добавляем их репозитории в ``sys.path`` (настраивается через
переменные окружения) и импортируем их, чтобы Forge всегда переиспользовал
реальные, поддерживаемые реализации.

Порядок разрешения для каждого репозитория:

1. Уже импортируемый (установлен / в PYTHONPATH)           -> используем как есть.
2. Переопределение через окружение (``DEVASSIST_PATH`` / ``EXTRACTORS_PATH``) -> добавляем в начало пути.
3. Привычное расположение рядом с этим репозиторием         -> добавляем в начало пути.

Каждый импортёр деградирует мягко: если репозиторий не найден, вызывающей
стороне сообщается об этом (``None``), и она переходит на офлайн-путь.
Ничего не падает при импорте.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import ModuleType

# Этот файл находится в <repo>/forge/integrations/_vendor.py; корень репозитория — на 3 уровня выше.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SIBLINGS = _REPO_ROOT.parent  # каталог, содержащий devassist/, test/ и т.п.


def _candidate_dirs(env_var: str, *sibling_names: str) -> list[Path]:
    dirs: list[Path] = []
    env = os.environ.get(env_var)
    if env:
        dirs.append(Path(env).expanduser())
    for name in sibling_names:
        dirs.append(_SIBLINGS / name)
    return dirs


def _import_from(
    module_name: str,
    env_var: str,
    *sibling_names: str,
) -> ModuleType | None:
    """Импортирует ``module_name``, при необходимости добавляя каталоги-кандидаты в ``sys.path``."""
    try:
        return importlib.import_module(module_name)
    except ImportError:
        pass

    top = module_name.split(".", 1)[0]
    for d in _candidate_dirs(env_var, *sibling_names):
        if not d.is_dir():
            continue
        # Пакет может находиться прямо в `d` (d/<top>/) — добавляем d в путь.
        if (d / top).is_dir() or any(d.glob(f"{top}*.egg-info")):
            if str(d) not in sys.path:
                sys.path.insert(0, str(d))
            try:
                return importlib.import_module(module_name)
            except ImportError:
                continue
    return None


def load_devassist_llm() -> ModuleType | None:
    """Возвращает пакет ``devassist.llm`` (types/base/gigachat) или ``None``."""
    return _import_from("devassist.llm", "DEVASSIST_PATH", "devassist")


def load_extractors() -> ModuleType | None:
    """Возвращает пакет ``extractors`` или ``None``.

    В эталонном репозитории содержащий каталог называется ``test`` (см. ссылки),
    поэтому мы проверяем как ``extractors`` (если перемещён), так и ``test``.
    """
    return _import_from("extractors", "EXTRACTORS_PATH", "extractors", "test")
