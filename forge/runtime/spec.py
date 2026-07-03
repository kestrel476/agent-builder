"""Спецификация среды выполнения и результаты для *сгенерированного* агента.

Упакованный агент на диске — это просто данные: промпт, контракт в виде
JSON-схемы и небольшая YAML-конфигурация с указанием его стратегии и порогов.
:class:`AgentSpec` загружает этот бандл; исполнитель его интерпретирует. Благодаря
этому сгенерированные агенты остаются декларативными и обозримыми — инженер
правит промпт/схему/конфигурацию, а не Python.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class AgentSpec:
    name: str
    archetype: str
    strategy: str                       # стратегия: extract | compare | generate | chat | workflow
    system_prompt: str
    output_schema: dict[str, Any]
    input_arity: int = 1
    confidence_threshold: float = 0.6
    rules: list[str] = field(default_factory=list)
    settings: dict[str, Any] = field(default_factory=dict)
    bundle_dir: Path | None = None

    @classmethod
    def load(cls, bundle_dir: str | Path) -> "AgentSpec":
        d = Path(bundle_dir)
        cfg = yaml.safe_load((d / "agent.yaml").read_text(encoding="utf-8")) or {}
        missing = [k for k in ("name", "archetype", "strategy") if not cfg.get(k)]
        if missing:
            raise ValueError(f"Некорректный agent.yaml в {d}: нет обязательных полей {missing}.")
        schema = json.loads((d / "contract.schema.json").read_text(encoding="utf-8"))
        prompt = (d / "prompt.md").read_text(encoding="utf-8")
        rules_path = d / "rules.json"
        rules = json.loads(rules_path.read_text(encoding="utf-8")) if rules_path.is_file() else []
        settings = dict(cfg.get("settings", {}))
        # Каталог правил rule-engine хранится отдельным файлом (может быть большим).
        rc_path = d / "rule_catalog.json"
        if rc_path.is_file():
            settings["rule_catalog"] = json.loads(rc_path.read_text(encoding="utf-8"))
        cl_path = d / "checklist.json"
        if cl_path.is_file():
            settings["checklist"] = json.loads(cl_path.read_text(encoding="utf-8"))
        return cls(
            name=cfg["name"],
            archetype=cfg["archetype"],
            strategy=cfg["strategy"],
            system_prompt=prompt,
            output_schema=schema,
            input_arity=int(cfg.get("input_arity", 1)),
            confidence_threshold=float(cfg.get("confidence_threshold", 0.6)),
            rules=list(rules),
            settings=settings,
            bundle_dir=d,
        )


@dataclass
class TraceStep:
    name: str
    detail: str = ""
    data: Any = None


@dataclass
class RunResult:
    """Итог запуска сгенерированного агента на одном входе."""

    status: str                          # статус: ok | partial | low_confidence | error
    output: Any = None                   # dict для JSON-агентов, str для документов/чата
    confidence: float | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    trace: list[TraceStep] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)  # наблюдаемость: вызовы LLM, токены

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "output": self.output,
            "confidence": self.confidence,
            "errors": self.errors,
            "warnings": self.warnings,
            "stats": self.stats,
            "trace": [{"name": s.name, "detail": s.detail} for s in self.trace],
        }
