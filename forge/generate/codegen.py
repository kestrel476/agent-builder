"""Синтез запускаемого *бандла* агента из чертежа.

Бандл намеренно сделан декларативным, чтобы инженер мог проверять и сопровождать
его, не читая сгенерированный Python:

    agent/
      agent.yaml            # стратегия, арность, пороги, возможности
      prompt.md             # системный промпт (редактируемый)
      contract.schema.json  # контракт вывода (единственный источник истины)
      rules.json            # бизнес-правила / проверки
      agent.py              # тонкая, стабильная точка входа (делегирует в forge.runtime)
      README.md             # как запускать, что ожидает на входе, что возвращает
      manifest.json         # происхождение: архетип, версия, возможности, хеш
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from ..archetypes import get as get_archetype
from ..model import AgentBlueprint
from ..store import Workspace

_AGENT_PY = '''#!/usr/bin/env python3
"""Точка входа для сгенерированного агента: {name}.

Сгенерировано Legal Agent Forge. Поведение задаётся редактируемыми файлами данных
рядом с этим скриптом (prompt.md, contract.schema.json, agent.yaml, rules.json),
а не этим кодом — чтобы изменить агента, правьте их.

Использование:
    python agent.py path/to/document.pdf
    python agent.py --text "сырой текст"
    echo '{{"k": "v"}}' | python agent.py --stdin
    python agent.py path/to/doc.pdf --debug --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Делаем агента запускаемым из исходников, даже если `legal-agent-forge` не
# установлен через pip: учитываем $FORGE_PATH, иначе считаем его импортируемым.
try:
    import forge  # noqa: F401
except ModuleNotFoundError:
    _fp = os.environ.get("FORGE_PATH")
    if _fp and _fp not in sys.path:
        sys.path.insert(0, _fp)

from forge.integrations import build_llm, DocumentReader
from forge.runtime import AgentSpec, AgentInput, run_agent

BUNDLE = Path(__file__).resolve().parent


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="{name} (сгенерированный юридический агент)")
    ap.add_argument("input", nargs="?", help="путь к документу или сырой текст")
    ap.add_argument("--text", help="трактовать аргумент как сырой текстовый ввод")
    ap.add_argument("--stdin", action="store_true", help="читать JSON/текст из stdin")
    ap.add_argument("--also", action="append", default=[], help="дополнительный входной файл (например, документ B для сравнения)")
    ap.add_argument("--offline", action="store_true", help="использовать детерминированный офлайн-мозг")
    ap.add_argument("--debug", action="store_true", help="вывести трассу выполнения")
    ap.add_argument("--dry-run", action="store_true", help="спланировать без вызова LLM")
    args = ap.parse_args(argv)

    spec = AgentSpec.load(BUNDLE)
    files = []
    text = None
    if args.stdin:
        raw = sys.stdin.read()
        try:
            text = None
            inp = AgentInput(json=json.loads(raw))
        except json.JSONDecodeError:
            inp = AgentInput(text=raw)
    else:
        if args.text is not None:
            inp = AgentInput(text=args.text)
        elif args.input and Path(args.input).exists():
            files = [args.input] + list(args.also)
            inp = AgentInput(files=files)
        elif args.input is not None:
            inp = AgentInput(text=args.input)
        else:
            ap.error("укажите входной файл, --text или --stdin")

    llm = build_llm(offline=args.offline)
    result = run_agent(spec, inp, llm=llm, reader=DocumentReader(), dry_run=args.dry_run)

    payload = result.to_dict()
    if not args.debug:
        payload.pop("trace", None)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if result.status in ("ok", "low_confidence", "partial") else 1


if __name__ == "__main__":
    raise SystemExit(main())
'''

_README = """# {name}

Сгенерировано **Legal Agent Forge** (архетип: `{archetype}`, стратегия: `{strategy}`).

## Что делает
{goal}

## Вход
{inputs}

## Выход
Возвращает JSON-конверт: `status` (ok / partial / low_confidence / error),
`output` (объект контракта ниже), `confidence`, `warnings` и `errors`.

## Запуск
```bash
python agent.py path/to/document.pdf          # реальный LLM (нужны учётные данные GigaChat)
python agent.py path/to/document.pdf --offline  # детерминированный офлайн-мозг
python agent.py --text "..." --debug --dry-run
```

## Контракт
См. `contract.schema.json`. Обязательные поля: {required}.

## Сопровождение
Правьте `prompt.md` (поведение), `contract.schema.json` (форма вывода),
`agent.yaml` (стратегия/пороги), `rules.json` (бизнес-правила). Python-точка
входа стабильна и редко требует изменений.

> Возможности: {capabilities}
"""


def synthesize_bundle(bp: AgentBlueprint, ws: Workspace) -> list[Path]:
    """Записать бандл агента в ``ws.agent_dir`` и вернуть пути к файлам."""
    arch = get_archetype(bp.archetype)
    bundle = ws.agent_dir
    bundle.mkdir(parents=True, exist_ok=True)
    schema = bp.io.output_schema()

    written: list[Path] = []

    def _write(name: str, content: str) -> None:
        p = bundle / name
        p.write_text(content, encoding="utf-8")
        written.append(p)

    _write("prompt.md", bp.effective_prompt() or _default_prompt(bp))
    _write("contract.schema.json", json.dumps(schema, ensure_ascii=False, indent=2))
    _write("rules.json", json.dumps(bp.instructions.business_rules, ensure_ascii=False, indent=2))

    settings: dict = {"output_kind": bp.io.output_kind.value}
    if bp.knowledge:
        settings["knowledge"] = _emit_knowledge(bp.knowledge, bundle)
    if bp.rule_catalog:
        (bundle / "rule_catalog.json").write_text(
            json.dumps(bp.rule_catalog, ensure_ascii=False, indent=2), encoding="utf-8")
        written.append(bundle / "rule_catalog.json")
    if bp.checklist:
        (bundle / "checklist.json").write_text(
            json.dumps(bp.checklist, ensure_ascii=False, indent=2), encoding="utf-8")
        written.append(bundle / "checklist.json")
    if bp.package:
        settings["package"] = bp.package
    config = {
        "name": bp.name,
        "archetype": bp.archetype,
        "strategy": arch.runtime_strategy,
        "input_arity": arch.input_arity,
        "confidence_threshold": 0.6 if bp.io.confidence_required else 0.0,
        "capabilities": [c.value for c in bp.capabilities],
        "settings": settings,
    }
    _write("agent.yaml", yaml.safe_dump(config, sort_keys=False, allow_unicode=True))

    _write("agent.py", _AGENT_PY.format(name=bp.name))

    required = ", ".join(f.name for f in bp.io.fields if f.required) or "(нет)"
    inputs = "\n".join(f"- `{i.name}` ({i.kind}{'' if i.required else ', опционально'})"
                       for i in bp.io.inputs) or "- (не указано)"
    _write("README.md", _README.format(
        name=bp.name, archetype=bp.archetype, strategy=arch.runtime_strategy,
        goal=bp.goal or "(см. спецификацию)", inputs=inputs, required=required,
        capabilities=", ".join(c.value for c in bp.capabilities)))

    manifest = {
        "name": bp.name,
        "slug": bp.slug,
        "archetype": bp.archetype,
        "version": bp.version,
        "strategy": arch.runtime_strategy,
        "capabilities": [c.value for c in bp.capabilities],
        "schema_hash": hashlib.sha256(
            json.dumps(schema, sort_keys=True).encode()).hexdigest()[:16],
        "open_issues": len(bp.open_issues),
        "assumptions": len(bp.assumptions),
        "test_cases": len(bp.test_cases),
    }
    _write("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    return written


def _emit_knowledge(kcfg: dict, bundle: Path) -> dict:
    """Скопировать каталог в бандл (knowledge/catalog.json) и вернуть конфиг для agent.yaml."""
    source = kcfg.get("source") or kcfg.get("file")
    kdir = bundle / "knowledge"
    kdir.mkdir(parents=True, exist_ok=True)
    dest = kdir / "catalog.json"
    if source and Path(source).is_file() and Path(source).resolve() != dest.resolve():
        dest.write_bytes(Path(source).read_bytes())
    out = {k: kcfg[k] for k in ("id_field", "title_field", "fact_field", "filter", "top_k") if k in kcfg}
    out["file"] = "knowledge/catalog.json"
    return out


def _default_prompt(bp: AgentBlueprint) -> str:
    return (
        f"Ты специализированный агент юридической автоматизации. Цель: {bp.goal or bp.name}. "
        "Работай только на основе предоставленного входа; если требуемое значение отсутствует, "
        "верни null и понизь уверенность. Никогда не выдумывай факты. Возвращай вывод строго согласно схеме."
    )
