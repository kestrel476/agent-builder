"""Терминальный интерфейс для Forge.

Небольшая обёртка вокруг rich, дающая CLI единый визуальный язык: панели, диалог
уточняющих вопросов, таблицы проблем/тестов и подтверждения. Сохраняется
независимым от остального кода, чтобы детали вывода не просачивались в логику
конвейера.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from rich.box import ROUNDED
from rich.console import Console as RichConsole
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

BRAND = "#7C9CF0"
ACCENT = "#22D3EE"
OK = "#34D399"
WARN = "#FBBF24"
DANGER = "#F87171"
MUTED = "#8A8A99"

_SEV_COLOR = {"blocker": DANGER, "major": WARN, "minor": ACCENT, "info": MUTED}
_STATUS_COLOR = {"pass": OK, "fail": DANGER, "error": DANGER, "skip": MUTED}


class Console:
    def __init__(self, *, no_color: bool = False, assume_yes: bool = False) -> None:
        self._c = RichConsole(no_color=no_color, highlight=False, emoji=False)
        self._assume_yes = assume_yes

    # -- примитивы -------------------------------------------------------- #
    def print(self, *a, **k) -> None:
        self._c.print(*a, **k)

    def blank(self) -> None:
        self._c.print()

    def rule(self, title: str = "") -> None:
        self._c.rule(Text(title, style=MUTED) if title else "", style=MUTED)

    def info(self, text: str) -> None:
        self._c.print(Text(text, style=ACCENT))

    def muted(self, text: str) -> None:
        self._c.print(Text(text, style=MUTED))

    def success(self, text: str) -> None:
        self._c.print(Text(f"✔ {text}", style=f"bold {OK}"))

    def warn(self, text: str) -> None:
        self._c.print(Text(f"⚠ {text}", style=WARN))

    def error(self, text: str) -> None:
        self._c.print(Text(f"✘ {text}", style=f"bold {DANGER}"))

    def step(self, text: str) -> None:
        self._c.print(Text("● ", style=BRAND) + Text(text, style="bold white"))

    # -- составные -------------------------------------------------------- #
    def banner(self, *, version: str, backend: str, model: str) -> None:
        logo = Text()
        logo.append("⚖ ", style=f"bold {BRAND}")
        logo.append("Legal Agent Forge", style="bold white")
        logo.append(f"  v{version}", style=MUTED)
        meta = Table.grid(padding=(0, 1))
        meta.add_column(style=MUTED, justify="right")
        meta.add_column()
        meta.add_row("мозг", Text(f"{backend} · {model}", style=ACCENT))
        meta.add_row("назначение", Text("сборка, тестирование и упаковка агентов юридической автоматизации", style="white"))
        self._c.print(Panel(Group(logo, Text(""), meta), box=ROUNDED, border_style=BRAND, padding=(1, 2), expand=False))

    def panel(self, body: str, *, title: str, border: str = MUTED) -> None:
        self._c.print(Panel(Text(body, style="white"), title=Text(title, style=ACCENT),
                            title_align="left", box=ROUNDED, border_style=border, padding=(0, 1), expand=False))

    def kv(self, rows: Sequence[tuple[str, str]], *, title: str | None = None) -> None:
        t = Table.grid(padding=(0, 2))
        t.add_column(style=MUTED, justify="right")
        t.add_column(style="white")
        for k, v in rows:
            t.add_row(k, str(v))
        if title:
            self._c.print(Panel(t, title=Text(title, style=ACCENT), title_align="left",
                               box=ROUNDED, border_style=MUTED, padding=(0, 1), expand=False))
        else:
            self._c.print(t)

    def issues_table(self, issues: Iterable) -> None:
        t = Table(box=ROUNDED, border_style=MUTED, header_style=f"bold {BRAND}", expand=False)
        t.add_column("id", style=MUTED, no_wrap=True)
        t.add_column("важн.")
        t.add_column("вид", style=ACCENT)
        t.add_column("статус", style=MUTED)
        t.add_column("сообщение", style="white", overflow="fold")
        any_row = False
        for i in issues:
            any_row = True
            sev = getattr(i, "severity", "info")
            sev = sev.value if hasattr(sev, "value") else str(sev)
            kind = getattr(i, "kind", "")
            kind = kind.value if hasattr(kind, "value") else str(kind)
            status = getattr(i, "status", "")
            status = status.value if hasattr(status, "value") else str(status)
            t.add_row(getattr(i, "id", "?"), Text(sev, style=_SEV_COLOR.get(sev, MUTED)),
                      kind, status, getattr(i, "message", ""))
        if any_row:
            self._c.print(t)
        else:
            self.muted("  (проблем нет)")

    def tests_table(self, rows: Iterable[dict]) -> None:
        t = Table(box=ROUNDED, border_style=MUTED, header_style=f"bold {BRAND}", expand=False)
        t.add_column("тест", style="white")
        t.add_column("статус")
        t.add_column("детали", style=MUTED, overflow="fold")
        for r in rows:
            st = r.get("status", "?")
            t.add_row(r.get("name", "?"), Text(st, style=_STATUS_COLOR.get(st, MUTED)), r.get("detail", ""))
        self._c.print(t)

    # -- взаимодействие --------------------------------------------------- #
    def ask(self, question: str, *, why: str | None = None, options: Sequence[str] | None = None,
            default: str | None = None) -> str:
        body = Text(question, style="white")
        if why:
            body.append(f"\n  ↳ зачем: {why}", style=MUTED)
        if options:
            body.append("\n  варианты: " + " | ".join(options), style=ACCENT)
        self._c.print(Panel(body, title=Text("уточнение", style=f"bold {BRAND}"), title_align="left",
                           box=ROUNDED, border_style=BRAND, padding=(0, 1), expand=False))
        prompt = Text(f"  ответ{f' [{default}]' if default else ''}: ", style=f"bold {ACCENT}")
        if self._assume_yes:
            self._c.print(prompt + Text(f"{default or '(пропущено)'} → авто", style=MUTED))
            return default or ""
        try:
            ans = self._c.input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            self._c.print()
            return default or ""
        return ans or (default or "")

    def confirm(self, question: str) -> bool:
        if self._assume_yes:
            return True
        try:
            ans = self._c.input(Text(f"  {question} [y/N] ", style=f"bold {WARN}")).strip().lower()
        except (EOFError, KeyboardInterrupt):
            self._c.print()
            return False
        return ans in ("y", "yes", "д", "да")
