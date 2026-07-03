"""Рабочая папка = один проект агента на диске.

Структура (внутри ``$FORGE_HOME``, по умолчанию ``./.forge``)::

    .forge/
      current                       # идентификатор(slug) активного проекта
      <slug>/
        blueprint.json              # IR (источник истины)
        spec.md                     # человекочитаемое представление спецификации
        inputs/                     # прикреплённые документы / примеры
        agent/                      # синтезированный, запускаемый бандл агента
        tests/cases/*.json          # опциональные вынесенные тест-кейсы
        runs/<n>/result.json        # по папке на каждый прогон тестов (наблюдаемость)
        CHANGELOG.md
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from ..config import ForgeConfig
from ..model import AgentBlueprint


# Транслитерация кириллицы в латиницу, чтобы русские имена давали
# осмысленные, файлово-безопасные slug'и (иначе они схлопывались бы в "agent").
_TRANSLIT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
    "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "h", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sch", "ъ": "",
    "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
}


def slugify(text: str) -> str:
    text = "".join(_TRANSLIT.get(ch, ch) for ch in text.strip().lower())
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text or "agent"


class Workspace:
    """Привязана к одной папке ``<slug>/``; загружает и сохраняет свой чертёж."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.slug = root.name

    # -- пути ------------------------------------------------------------- #
    @property
    def blueprint_path(self) -> Path:
        return self.root / "blueprint.json"

    @property
    def spec_path(self) -> Path:
        return self.root / "spec.md"

    @property
    def inputs_dir(self) -> Path:
        return self.root / "inputs"

    @property
    def agent_dir(self) -> Path:
        return self.root / "agent"

    @property
    def tests_dir(self) -> Path:
        return self.root / "tests"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    @property
    def changelog_path(self) -> Path:
        return self.root / "CHANGELOG.md"

    # -- ввод-вывод чертежа ----------------------------------------------- #
    def exists(self) -> bool:
        return self.blueprint_path.is_file()

    def load(self) -> AgentBlueprint:
        data = json.loads(self.blueprint_path.read_text(encoding="utf-8"))
        return AgentBlueprint.model_validate(data)

    def save(self, bp: AgentBlueprint) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.blueprint_path.write_text(
            json.dumps(bp.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # -- вложения --------------------------------------------------------- #
    def attach(self, src: Path) -> Path:
        """Копирует вложение в ``inputs/`` и возвращает сохранённый путь."""
        self.inputs_dir.mkdir(parents=True, exist_ok=True)
        dest = self.inputs_dir / src.name
        dest.write_bytes(Path(src).read_bytes())
        return dest

    # -- прогоны тестов --------------------------------------------------- #
    def new_run_dir(self) -> Path:
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        n = 1 + max((int(p.name) for p in self.runs_dir.glob("*") if p.name.isdigit()), default=0)
        d = self.runs_dir / str(n)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def latest_run_dir(self) -> Path | None:
        if not self.runs_dir.is_dir():
            return None
        runs = [p for p in self.runs_dir.glob("*") if p.name.isdigit()]
        return max(runs, key=lambda p: int(p.name), default=None)

    # -- журнал изменений ------------------------------------------------- #
    def append_changelog(self, line: str) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        prev = self.changelog_path.read_text(encoding="utf-8") if self.changelog_path.is_file() else "# Журнал изменений\n"
        self.changelog_path.write_text(f"{prev}\n- {stamp} — {line}", encoding="utf-8")


class WorkspaceManager:
    """Создаёт / открывает / перечисляет рабочие папки и отслеживает текущую."""

    def __init__(self, config: ForgeConfig) -> None:
        self.config = config
        self.home = config.ensure_home()

    @property
    def _current_pointer(self) -> Path:
        return self.home / "current"

    def create(self, name: str) -> Workspace:
        slug = self._unique_slug(slugify(name))
        ws = Workspace(self.home / slug)
        ws.root.mkdir(parents=True, exist_ok=True)
        self.set_current(slug)
        return ws

    def open(self, slug: str) -> Workspace:
        ws = Workspace(self.home / slug)
        if not ws.exists():
            raise FileNotFoundError(f"Нет рабочей папки '{slug}' в {self.home}")
        return ws

    def current(self) -> Workspace | None:
        if not self._current_pointer.is_file():
            return None
        slug = self._current_pointer.read_text(encoding="utf-8").strip()
        ws = Workspace(self.home / slug)
        return ws if ws.exists() else None

    def require_current(self) -> Workspace:
        ws = self.current()
        if ws is None:
            raise FileNotFoundError(
                "Нет активного агента. Создайте его командой `forge new \"...\"` или выберите командой `forge use <slug>`."
            )
        return ws

    def set_current(self, slug: str) -> None:
        self._current_pointer.write_text(slug, encoding="utf-8")

    def list(self) -> list[Workspace]:
        out = []
        for p in sorted(self.home.glob("*")):
            if p.is_dir() and (p / "blueprint.json").is_file():
                out.append(Workspace(p))
        return out

    def _unique_slug(self, base: str) -> str:
        slug, n = base, 1
        while (self.home / slug).exists():
            n += 1
            slug = f"{base}-{n}"
        return slug
