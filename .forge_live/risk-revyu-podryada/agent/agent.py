#!/usr/bin/env python3
"""Точка входа для сгенерированного агента: Риск-ревью подряда.

Сгенерировано Legal Agent Forge. Поведение задаётся редактируемыми файлами данных
рядом с этим скриптом (prompt.md, contract.schema.json, agent.yaml, rules.json),
а не этим кодом — чтобы изменить агента, правьте их.

Использование:
    python agent.py path/to/document.pdf
    python agent.py --text "сырой текст"
    echo '{"k": "v"}' | python agent.py --stdin
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
    ap = argparse.ArgumentParser(description="Риск-ревью подряда (сгенерированный юридический агент)")
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
