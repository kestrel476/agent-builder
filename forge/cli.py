"""Интерфейс командной строки для Legal Agent Forge.

Два способа управления:

* **Команды этапов** — `new`, `clarify`, `contract`, `build`, `test`, `package`,
  `run`, `status`, … для точного, скриптуемого контроля.
* **`wizard`** — мастер (пошаговый режим), проводящий через весь цикл для одного агента.

Глобальные флаги (перед подкомандой): `--offline`, `--model`, `--yes`,
`--no-color`, `--home DIR`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .archetypes import all_archetypes
from .config import ForgeConfig
from .console import Console
from .model import AgentBlueprint, FieldSpec, TestCase
from .pipeline import Forge
from .store import Workspace, WorkspaceManager


class CLI:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.config = ForgeConfig.load(
            home=args.home, offline=args.offline, model=args.model,
            assume_yes=args.yes, no_color=args.no_color,
        )
        self.console = Console(no_color=args.no_color, assume_yes=args.yes)
        self.wm = WorkspaceManager(self.config)
        self._forge: Forge | None = None

    @property
    def forge(self) -> Forge:
        if self._forge is None:
            self._forge = Forge(offline=self.config.offline, model=self.config.model)
        return self._forge

    # ------------------------------------------------------------------ #
    # Вспомогательные методы
    # ------------------------------------------------------------------ #
    def _current(self) -> tuple[Workspace, AgentBlueprint]:
        ws = self.wm.require_current()
        return ws, ws.load()

    def _save(self, ws: Workspace, bp: AgentBlueprint) -> None:
        ws.save(bp)

    def _gather_instructions(self) -> str:
        parts: list[str] = []
        if self.args.description:
            parts.append(" ".join(self.args.description))
        for f in getattr(self.args, "from_files", []) or []:
            p = Path(f)
            if not p.exists():
                self.console.warn(f"файл с инструкциями не найден: {f}")
                continue
            res = self.forge.reader.read(p)
            if res.text:
                parts.append(f"\n[из {p.name}]\n{res.text}")
            else:
                self.console.warn(f"не удалось прочитать {f}: {res.error or res.status}")
        return "\n\n".join(parts).strip()

    # ------------------------------------------------------------------ #
    # Команды
    # ------------------------------------------------------------------ #
    def cmd_new(self) -> int:
        raw = self._gather_instructions()
        if not raw:
            self.console.error("Укажите описание и/или --from ФАЙЛ.")
            return 2
        name = self.args.name or _name_from_text(raw)
        archetype = self.args.archetype
        if not archetype:
            key, conf, why = self.forge.detect_archetype(raw)
            archetype = key
            self.console.muted(f"определён архетип: {key} ({conf:.0%}) — {why}")
        ws = self.wm.create(name)
        bp = AgentBlueprint(name=name, slug=ws.slug, archetype=archetype, instructions_raw=raw)
        bp.record("new", f"archetype={archetype}")

        self.console.banner(version=__version__, backend=self.forge.llm.backend_name, model=self.forge.llm.model)
        self.console.step(f"Создан агент '{name}' [{ws.slug}] как {archetype}")

        # Этап 0 + анализ уточнения сразу же.
        self.forge.intake(bp)
        self.forge.analyze(bp, use_llm=not self.config.offline)
        self._attach_files(ws)
        self._save(ws, bp)
        self._print_intake_summary(bp)
        self.console.muted("далее: `forge clarify`, чтобы разрешить открытые вопросы, или `forge wizard`, чтобы пройти весь цикл.")
        return 0

    def _attach_files(self, ws: Workspace) -> None:
        for f in getattr(self.args, "attach", []) or []:
            p = Path(f)
            if p.exists():
                ws.attach(p)
                self.console.muted(f"приложен вход: {p.name}")

    def cmd_clarify(self) -> int:
        ws, bp = self._current()
        self.forge.analyze(bp, use_llm=not self.config.offline)
        pending = self.forge.clarify.pending_questions(bp)
        if not pending:
            self.console.success("Открытых вопросов нет. Контракт разблокирован.")
            self._save(ws, bp)
            return 0
        if self.config.assume_yes or self.args.auto:
            made = self.forge.assume_open(bp, only_blocking=self.args.blocking_only)
            self.console.warn(f"--auto: записано явных допущений ({len(made)}) вместо вопросов:")
            for a in made:
                self.console.muted(f"  • [{a.risk.value}] {a.statement}")
            self._save(ws, bp)
            return 0

        self.console.step(f"Открытых вопросов: {len(pending)}. Нажмите Enter, чтобы пропустить (вопрос станет допущением).")
        asked = 0
        for q in pending:
            if self.args.max and asked >= self.args.max:
                break
            ans = self.console.ask(q.text, why=q.why, options=q.options or None)
            if ans.strip():
                self.forge.answer(bp, q.id, ans)
            asked += 1
        # Всё, что осталось открытым после раунда → допущения, записанные явно.
        remaining = self.forge.clarify.pending_questions(bp)
        if remaining:
            made = self.forge.assume_open(bp, only_blocking=False)
            self.console.muted(f"записано допущений ({len(made)}) для оставшихся без ответа вопросов.")
        self._save(ws, bp)
        self.console.success("Раунд уточнения завершён.")
        return 0

    def cmd_contract(self) -> int:
        ws, bp = self._current()
        self.forge.build_contract(bp)
        self._save(ws, bp)
        self._print_contract(bp)
        if bp.blocking_issues:
            self.console.warn(f"Блокирующих замечаний осталось: {len(bp.blocking_issues)} — запустите `forge clarify`.")
        else:
            self.console.success("Контракт зафиксирован. Готов к сборке.")
        return 0

    def cmd_field(self) -> int:
        ws, bp = self._current()
        if self.args.add:
            by_name = {f.name: i for i, f in enumerate(bp.io.fields)}
            added = updated = 0
            for spec in self.args.add:
                field = _parse_field(spec)
                if field.name in by_name:  # upsert: заменить поле с тем же именем
                    bp.io.fields[by_name[field.name]] = field
                    updated += 1
                else:
                    by_name[field.name] = len(bp.io.fields)
                    bp.io.fields.append(field)
                    added += 1
            self.console.success(f"добавлено полей: {added}, обновлено: {updated}")
        if self.args.remove:
            before = len(bp.io.fields)
            bp.io.fields = [f for f in bp.io.fields if f.name not in self.args.remove]
            self.console.success(f"удалено полей: {before - len(bp.io.fields)}")
        self._save(ws, bp)
        self._print_contract(bp)
        return 0

    def cmd_knowledge(self) -> int:
        ws, bp = self._current()
        src = Path(self.args.file)
        if not src.is_file():
            self.console.error(f"Файл каталога не найден: {src}")
            return 2
        dest = ws.attach(src)  # копия в inputs/ для самодостаточности
        filt: dict[str, str] = {}
        for f in (self.args.filter or []):
            k, _, v = f.partition("=")
            if k.strip():
                filt[k.strip()] = v.strip()
        bp.knowledge = {
            "source": str(dest),
            "id_field": self.args.id_field or "__key__",
            "title_field": self.args.title_field or "title",
            "fact_field": self.args.fact_field or "fact",
            "filter": filt,
            "top_k": self.args.top_k or 30,
        }
        bp.record("knowledge", f"каталог {src.name}, фильтр={filt or '—'}")
        self._save(ws, bp)
        # быстрый подсчёт размера подмножества
        try:
            data = json.loads(dest.read_text(encoding="utf-8"))
            items = data.values() if isinstance(data, dict) else data
            n = sum(1 for o in items if isinstance(o, dict) and all(o.get(k) == v for k, v in filt.items()))
            total = len(data)
            self.console.success(f"База знаний привязана: {src.name} ({n}/{total} записей после фильтра)")
        except Exception:
            self.console.success(f"База знаний привязана: {src.name}")
        return 0

    def cmd_rules(self) -> int:
        ws, bp = self._current()
        src = Path(self.args.file)
        if not src.is_file():
            self.console.error(f"Файл правил не найден: {src}")
            return 2
        data = json.loads(src.read_text(encoding="utf-8"))
        rules = data if isinstance(data, list) else data.get("rules", [])
        rules = [r for r in rules if isinstance(r, dict) and r.get("id")]
        bp.rule_catalog = rules
        bp.record("rules", f"каталог правил: {len(rules)}")
        self._save(ws, bp)
        self.console.success(f"Каталог правил привязан: {len(rules)} правил(а) из {src.name}")
        return 0

    def cmd_checklist(self) -> int:
        ws, bp = self._current()
        src = Path(self.args.file)
        if not src.is_file():
            self.console.error(f"Файл чек-листа не найден: {src}")
            return 2
        data = json.loads(src.read_text(encoding="utf-8"))
        checks = data if isinstance(data, list) else data.get("checks", [])
        checks = [c for c in checks if isinstance(c, dict) and c.get("id")]
        bp.checklist = checks
        bp.record("checklist", f"проверок: {len(checks)}")
        self._save(ws, bp)
        self.console.success(f"Чек-лист привязан: {len(checks)} проверок из {src.name}")
        return 0

    def cmd_routing(self) -> int:
        ws, bp = self._current()
        cfg: dict = {}
        if self.args.file:
            p = Path(self.args.file)
            if not p.is_file():
                self.console.error(f"Файл маршрутизации не найден: {p}")
                return 2
            cfg = json.loads(p.read_text(encoding="utf-8"))
        package = {
            "taxonomy": cfg.get("taxonomy", []),
            "required": cfg.get("required", []),
            "per_type": cfg.get("per_type", {}),
        }
        if self.args.required:
            package["required"] = [c.strip() for c in self.args.required.split(",") if c.strip()]
        bp.package = package
        bp.record("routing", f"типов: {len(package['taxonomy'])}, обязательных: {len(package['required'])}")
        self._save(ws, bp)
        self.console.success(
            f"Маршрутизация пакета задана: {len(package['taxonomy'])} тип(ов), "
            f"обязательные: {package['required'] or '—'}")
        return 0

    def cmd_build(self) -> int:
        ws, bp = self._current()
        if bp.blocking_issues:
            self.console.warn(f"Открытых блокирующих замечаний: {len(bp.blocking_issues)}. "
                              "Сборка несмотря на это зафиксирует их в отчёте о передаче инженеру.")
            if not self.console.confirm("Собрать несмотря на блокирующие замечания?"):
                return 1
        if not bp.io.fields and bp.io.output_kind.value in ("json", "classification", "report"):
            self.forge.build_contract(bp)
        written = self.forge.synthesize(bp, ws)
        self._save(ws, bp)
        self.console.success(f"Синтезирован бандл агента (файлов: {len(written)}) → {ws.agent_dir}")
        for p in sorted(written):
            self.console.muted(f"  {p.relative_to(ws.root)}")
        return 0

    def cmd_addtest(self) -> int:
        ws, bp = self._current()
        also_file = getattr(self.args, "also_file", None) or []
        input_files = ([self.args.file, *also_file] if self.args.file and also_file else [])
        tc = TestCase(
            id=bp.next_id("T"),
            name=self.args.name or f"case-{len(bp.test_cases)+1}",
            input_text=self.args.text,
            input_file=self.args.file if not input_files else None,
            input_files=input_files,
            input_json=json.loads(self.args.json) if self.args.json else None,
            expected=json.loads(self.args.expect) if self.args.expect else None,
            must_contain=self.args.contains or [],
            metric=json.loads(self.args.metric) if self.args.metric else None,
        )
        bp.test_cases.append(tc)
        self._save(ws, bp)
        self.console.success(f"добавлен тест '{tc.name}' [{tc.id}] (всего: {len(bp.test_cases)})")
        return 0

    def cmd_test(self) -> int:
        ws, bp = self._current()
        if not bp.test_cases:
            self.console.warn("Нет тест-кейсов. Добавьте через `forge addtest --name x --text '...' --expect '{...}'`.")
            return 1
        if self.args.refine:
            history = self.forge.test_and_refine(bp, ws, max_iters=self.args.max or 3)
            for i, report, diagnoses in history:
                self.console.step(f"итерация {i}: пройдено {report.passed}/{report.total}")
                self.console.tests_table(report.rows())
                for d in diagnoses:
                    self.console.panel(f"{d.get('root_cause','')}\n→ {d.get('fix_suggestion','')}",
                                       title=f"диагноз: {d.get('case','?')}")
            last = history[-1][1]
            self._save(ws, bp)
            (self.console.success if last.green else self.console.warn)(
                f"итог: пройдено {last.passed}/{last.total} (v{bp.version})")
            return 0 if last.green else 1
        report = self.forge.test(bp, ws, only=self.args.case)
        self._save(ws, bp)
        self.console.tests_table(report.rows())
        for r in report.failed:
            for d in r.diffs:
                self.console.muted(f"  {r.name} · {d['field']}: ожидалось={d['expected']!r} получено={d['actual']!r}")
        (self.console.success if report.green else self.console.warn)(
            f"пройдено {report.passed}/{report.total} → прогон {report.run_dir.name if report.run_dir else '-'}")
        return 0 if report.green else 1

    def cmd_package(self) -> int:
        ws, bp = self._current()
        report = None
        if bp.test_cases:
            report = self.forge.test(bp, ws)
        self.forge.package(bp, ws, report)
        self._save(ws, bp)
        self.console.success(f"Упакована версия v{bp.version} → {ws.root}")
        self.console.kv([
            ("spec", str(ws.spec_path.relative_to(ws.root))),
            ("передача инженеру", "HANDOFF.md"),
            ("бандл", str(ws.agent_dir.relative_to(ws.root)) + "/"),
            ("changelog", "CHANGELOG.md"),
        ], title="артефакты")
        if bp.assumptions:
            self.console.warn(f"Записано допущений: {len(bp.assumptions)} — просмотрите HANDOFF.md перед продакшеном.")
        return 0

    def cmd_run(self) -> int:
        ws, bp = self._current()
        if not ws.agent_dir.joinpath("agent.yaml").is_file():
            self.console.error("Нет бандла. Сначала запустите `forge build`.")
            return 2
        from .runtime import AgentInput
        also = getattr(self.args, "also", None) or []
        # Явный --text — это всегда текст; не угадываем путь vs текст.
        if self.args.text is not None:
            value = AgentInput(text=self.args.text)
        elif self.args.input is None:
            self.console.error("Укажите входной файл или --text.")
            return 2
        elif also or Path(self.args.input).is_dir():
            # Несколько документов (для compare/package): первый + --also.
            value = AgentInput(files=[self.args.input, *also])
        else:
            value = self.args.input  # путь или текст — разрешит AgentInput.from_value
        result = self.forge.run_agent_bundle(ws, value, dry_run=self.args.dry_run)
        payload = result.to_dict()
        if not self.args.debug:
            payload.pop("trace", None)
        # Печатаем напрямую в stdout (не через rich), чтобы вывод был валидным
        # машиночитаемым JSON при перенаправлении в файл/конвейер.
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0 if result.status in ("ok", "low_confidence", "partial") else 1

    def cmd_status(self) -> int:
        ws, bp = self._current()
        self.console.banner(version=__version__, backend=self.forge.llm.backend_name, model=self.forge.llm.model)
        self.console.kv([
            ("агент", f"{bp.name} [{bp.slug}]"),
            ("архетип", bp.archetype),
            ("этап", bp.stage.value),
            ("версия", str(bp.version)),
            ("возможности", ", ".join(c.value for c in bp.capabilities) or "-"),
            ("открытые вопросы", str(len(bp.open_questions))),
            ("допущения", str(len(bp.assumptions))),
            ("тест-кейсы", str(len(bp.test_cases))),
        ], title="агент")
        if bp.open_issues:
            self.console.print("")
            self.console.issues_table(bp.open_issues)
        latest = ws.latest_run_dir()
        if latest:
            data = json.loads((latest / "result.json").read_text())
            self.console.muted(f"последний прогон тестов #{latest.name}: пройдено {data['passed']}/{data['total']}")
        return 0

    def cmd_show(self) -> int:
        ws, bp = self._current()
        what = self.args.what
        if what == "spec":
            self.console.print(ws.spec_path.read_text() if ws.spec_path.is_file()
                               else _lazy_spec(bp))
        elif what == "blueprint":
            self.console.print(json.dumps(bp.model_dump(mode="json"), ensure_ascii=False, indent=2))
        elif what == "prompt":
            self.console.print(bp.effective_prompt() or "(ещё не синтезирован)")
        elif what == "contract":
            self._print_contract(bp)
        elif what == "issues":
            self.console.issues_table(bp.issues)
        elif what == "questions":
            for q in bp.questions:
                mark = "✓" if q.answered else ("!" if q.blocking else "·")
                self.console.print(f"[{mark}] {q.id} {q.text}")
                if q.answered:
                    self.console.muted(f"      → {q.answer}")
        elif what == "assumptions":
            for a in bp.assumptions:
                self.console.print(f"[{a.risk.value}] {a.id}: {a.statement}")
        return 0

    def cmd_list(self) -> int:
        rows = []
        cur = self.wm.current()
        cur_slug = cur.slug if cur else None
        for ws in self.wm.list():
            bp = ws.load()
            mark = "→" if ws.slug == cur_slug else " "
            rows.append((f"{mark} {ws.slug}", f"{bp.archetype} · {bp.stage.value} · v{bp.version}"))
        if not rows:
            self.console.muted("Пока нет агентов. Создайте: `forge new \"...\"`.")
            return 0
        self.console.kv(rows, title="агенты")
        return 0

    def cmd_use(self) -> int:
        self.wm.open(self.args.slug)  # проверяет существование
        self.wm.set_current(self.args.slug)
        self.console.success(f"текущий агент → {self.args.slug}")
        return 0

    def cmd_archetypes(self) -> int:
        for a in all_archetypes():
            self.console.print(f"• {a.key} — {a.title}")
            self.console.muted(f"    {a.description}")
        return 0

    def cmd_wizard(self) -> int:
        """Пошаговый сквозной цикл для одного агента."""
        if self.args.description or self.args.from_files:
            rc = self.cmd_new()
            if rc != 0:
                return rc
        ws, bp = self._current()
        self.console.rule("уточнение")
        self.args.auto = self.args.auto if hasattr(self.args, "auto") else False
        self.args.blocking_only = False
        self.args.max = None
        self.cmd_clarify()
        self.console.rule("контракт")
        self.cmd_contract()
        self.console.rule("сборка")
        self.cmd_build()
        ws, bp = self._current()
        if bp.test_cases:
            self.console.rule("тест + доработка")
            self.args.refine = True
            self.args.case = None
            self.cmd_test()
        else:
            self.console.muted("нет тест-кейсов — пропускаем этап тестирования (добавьте их через `forge addtest`).")
        self.console.rule("упаковка")
        self.cmd_package()
        self.console.success("Мастер завершён.")
        return 0

    # ------------------------------------------------------------------ #
    # Форматированный вывод
    # ------------------------------------------------------------------ #
    def _print_intake_summary(self, bp: AgentBlueprint) -> None:
        ins = bp.instructions
        self.console.kv([
            ("цель", bp.goal or "-"),
            ("правила", str(len(ins.business_rules))),
            ("неоднозначности", str(len(ins.ambiguities))),
            ("отсутствующие данные", str(len(ins.missing_data))),
            ("открытые вопросы", str(len(bp.open_questions))),
            ("блокирующие", str(len(bp.blocking_issues))),
        ], title="приём (intake)")
        if ins.ambiguities:
            self.console.warn("Отмечены неоднозначности (без догадок):")
            for a in ins.ambiguities[:6]:
                self.console.muted(f"  • {a}")

    def _print_contract(self, bp: AgentBlueprint) -> None:
        rows = [(f.name, f"{f.type}{' *' if f.required else ''}  {f.description}") for f in bp.io.fields]
        if rows:
            self.console.kv(rows, title="поля вывода (* = обязательное)")
        self.console.kv(
            [(i.name, f"{i.kind}{'' if i.required else ' (необязательный)'}") for i in bp.io.inputs],
            title="входы")
        if bp.io.success_criteria:
            self.console.panel("\n".join(f"- {c}" for c in bp.io.success_criteria), title="критерии успеха")


# --------------------------------------------------------------------------- #
# Вспомогательные функции
# --------------------------------------------------------------------------- #
def _name_from_text(text: str) -> str:
    first = text.strip().splitlines()[0]
    return (first[:48].rstrip(" .,:;") or "Legal Agent")


def _parse_field(spec: str) -> FieldSpec:
    # name:type:required:description
    parts = spec.split(":", 3)
    name = parts[0].strip()
    typ = parts[1].strip() if len(parts) > 1 and parts[1] else "string"
    required = (len(parts) > 2 and parts[2].strip().lower() in ("1", "true", "yes", "req", "required"))
    desc = parts[3].strip() if len(parts) > 3 else ""
    return FieldSpec(name=name, type=typ, required=required, description=desc)


def _lazy_spec(bp: AgentBlueprint) -> str:
    from .pipeline.packaging import render_spec
    return render_spec(bp)


# --------------------------------------------------------------------------- #
# Парсер аргументов
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="forge", description="Legal Agent Forge — сборка агентов для автоматизации юридических процессов.")
    p.add_argument("--version", action="version", version=f"legal-agent-forge {__version__}")
    p.add_argument("--offline", action="store_true", help="использовать детерминированный офлайн-движок (без учётных данных LLM)")
    p.add_argument("--model", help="переопределить модель GigaChat")
    p.add_argument("--yes", "-y", action="store_true", help="неинтерактивный режим: автоматически записывать допущения")
    p.add_argument("--no-color", action="store_true")
    p.add_argument("--home", help="корень рабочей папки (по умолчанию ./.forge или $FORGE_HOME)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("new", help="создать нового агента из описания / файлов с инструкциями")
    sp.add_argument("description", nargs="*", help="произвольное текстовое описание агента")
    sp.add_argument("--from", dest="from_files", action="append", default=[], help="файл(ы) с инструкциями")
    sp.add_argument("--attach", action="append", default=[], help="пример(ы) документов для сохранения как входы")
    sp.add_argument("--archetype", help="принудительно задать архетип (см. `forge archetypes`)")
    sp.add_argument("--name", help="человекочитаемое имя агента")

    sp = sub.add_parser("clarify", help="разрешить открытые вопросы (интерактивно или --auto)")
    sp.add_argument("--auto", action="store_true", help="записывать допущения вместо вопросов")
    sp.add_argument("--blocking-only", action="store_true", help="с --auto допускать только блокирующие вопросы")
    sp.add_argument("--max", type=int, help="задать не более N вопросов за раунд")

    sub.add_parser("contract", help="построить/просмотреть контракт ввода-вывода")

    sp = sub.add_parser("field", help="редактировать поля вывода")
    sp.add_argument("--add", action="append", help="имя:тип:обязательное:описание")
    sp.add_argument("--remove", action="append", help="имя поля для удаления")

    sp = sub.add_parser("knowledge", help="привязать каталог-базу знаний (для RAG-архетипов)")
    sp.add_argument("--file", required=True, help="JSON-каталог (словарь код→запись или список)")
    sp.add_argument("--filter", action="append", help="поле=значение — ограничить подмножеством каталога")
    sp.add_argument("--id-field", dest="id_field", help="поле-идентификатор (по умолчанию ключ словаря)")
    sp.add_argument("--title-field", dest="title_field", help="поле названия (по умолчанию title)")
    sp.add_argument("--fact-field", dest="fact_field", help="поле типового факта (по умолчанию fact)")
    sp.add_argument("--top-k", dest="top_k", type=int, help="кандидатов на обстоятельство (по умолчанию 30)")

    sp = sub.add_parser("rules", help="привязать каталог правил (для rule_check)")
    sp.add_argument("--file", required=True, help="JSON со списком правил (id, match, terms/pattern, ...)")

    sp = sub.add_parser("checklist", help="привязать чек-лист проверок (для checklist)")
    sp.add_argument("--file", required=True, help="JSON со списком проверок (id, question, criteria, on_yes_*)")

    sp = sub.add_parser("routing", help="задать маршрутизацию пакета (для document_package)")
    sp.add_argument("--file", help="JSON {taxonomy, required, per_type}")
    sp.add_argument("--required", help="коды обязательных типов через запятую")

    sub.add_parser("build", help="синтезировать исполняемый бандл агента")

    sp = sub.add_parser("addtest", help="добавить тест-кейс")
    sp.add_argument("--name")
    sp.add_argument("--text", help="встроенный текстовый вход")
    sp.add_argument("--file", help="документ-вход (относительно inputs/)")
    sp.add_argument("--also-file", dest="also_file", action="append",
                    help="ещё документ-вход (для compare/package)")
    sp.add_argument("--json", help="встроенный JSON-вход")
    sp.add_argument("--expect", help="ожидаемый объект вывода в формате JSON")
    sp.add_argument("--contains", action="append", help="подстрока, которую должен содержать вывод")
    sp.add_argument("--metric", help='метрика P/R/F1 для list-поля, JSON: '
                    '{"field":"detected_risks","key":"risk_code","expected":[...],"min_recall":1.0}')

    sp = sub.add_parser("test", help="запустить тесты; --refine для цикла до зелёного результата")
    sp.add_argument("--case", help="запустить только этот кейс по id/имени")
    sp.add_argument("--refine", action="store_true", help="диагностировать сбои и дорабатывать, итеративно")
    sp.add_argument("--max", type=int, help="макс. число итераций доработки (по умолчанию 3)")

    sub.add_parser("package", help="создать итоговый spec + передачу инженеру + бандл")

    sp = sub.add_parser("run", help="запустить текущего агента на входе")
    sp.add_argument("input", nargs="?", help="путь к документу/директории-пакету или текст")
    sp.add_argument("--text", help="трактовать как сырой текст")
    sp.add_argument("--also", action="append", help="ещё документ (напр. документ B для сравнения)")
    sp.add_argument("--dry-run", action="store_true")
    sp.add_argument("--debug", action="store_true", help="включить трассировку выполнения")

    sub.add_parser("status", help="наблюдаемость для текущего агента")

    sp = sub.add_parser("show", help="вывести артефакт")
    sp.add_argument("what", choices=["spec", "blueprint", "prompt", "contract", "issues", "questions", "assumptions"])

    sub.add_parser("list", help="список агентов")
    sp = sub.add_parser("use", help="выбрать текущего агента")
    sp.add_argument("slug")
    sub.add_parser("archetypes", help="список доступных архетипов")

    sp = sub.add_parser("wizard", help="пошаговый сквозной цикл")
    sp.add_argument("description", nargs="*")
    sp.add_argument("--from", dest="from_files", action="append", default=[])
    sp.add_argument("--attach", action="append", default=[])
    sp.add_argument("--archetype")
    sp.add_argument("--name")
    sp.add_argument("--auto", action="store_true")
    return p


_DISPATCH = {
    "new": "cmd_new", "clarify": "cmd_clarify", "contract": "cmd_contract", "field": "cmd_field",
    "knowledge": "cmd_knowledge", "rules": "cmd_rules", "routing": "cmd_routing", "checklist": "cmd_checklist",
    "build": "cmd_build", "addtest": "cmd_addtest", "test": "cmd_test", "package": "cmd_package",
    "run": "cmd_run", "status": "cmd_status", "show": "cmd_show", "list": "cmd_list",
    "use": "cmd_use", "archetypes": "cmd_archetypes", "wizard": "cmd_wizard",
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # нормализуем необязательные атрибуты, используемые в разных командах
    for attr, default in (("from_files", []), ("attach", []), ("description", []),
                          ("auto", False), ("refine", False), ("case", None), ("max", None),
                          ("name", None), ("archetype", None)):
        if not hasattr(args, attr):
            setattr(args, attr, default)
    cli = CLI(args)
    method = getattr(cli, _DISPATCH[args.cmd])
    try:
        return method()
    except FileNotFoundError as e:
        cli.console.error(str(e))
        return 2
    except KeyboardInterrupt:
        cli.console.error("прервано")
        return 130


if __name__ == "__main__":
    sys.exit(main())
