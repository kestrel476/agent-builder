"""Небольшой валидатор JSON-схем без внешних зависимостей.

Сгенерированные агенты валидируют свой собственный вывод по схеме контракта. Мы
избегаем подключения ``jsonschema``, чтобы у упакованного агента была минимальная
поверхность зависимостей. Поддерживается то подмножество, которое Forge реально
выдаёт: object/array/string/number/integer/boolean, ``required``, ``enum``,
``additionalProperties: false`` и вложенные ``items``/``properties``.
"""

from __future__ import annotations

from typing import Any

_TYPE_OK = {
    "string": lambda v: isinstance(v, str),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
    "null": lambda v: v is None,
}


def validate(schema: dict[str, Any], value: Any, *, path: str = "$") -> list[str]:
    """Вернуть список понятных человеку ошибок валидации ([], если всё корректно)."""
    errors: list[str] = []
    _validate(schema, value, path, errors)
    return errors


def _enum_ok(value: Any, allowed: list) -> bool:
    """Сверка с перечислением; для строк — без учёта регистра и крайних пробелов."""
    if isinstance(value, str):
        norm = value.strip().lower()
        return any(isinstance(a, str) and a.strip().lower() == norm for a in allowed)
    return value in allowed


def _validate(schema: dict, value: Any, path: str, errors: list[str]) -> None:
    typ = schema.get("type")
    # Разрешаем null для необязательных значений, вернувшихся как None.
    if value is None:
        return
    if typ and typ in _TYPE_OK and not _TYPE_OK[typ](value):
        errors.append(f"{path}: ожидался тип {typ}, получен {type(value).__name__}")
        return
    if "enum" in schema and not _enum_ok(value, schema["enum"]):
        errors.append(f"{path}: {value!r} отсутствует в перечислении {schema['enum']}")
    if typ == "object":
        props = schema.get("properties", {})
        for req in schema.get("required", []):
            if req not in value or value[req] in (None, ""):
                errors.append(f"{path}.{req}: обязательное поле отсутствует или пусто")
        if schema.get("additionalProperties") is False:
            for k in value:
                if k not in props:
                    errors.append(f"{path}.{k}: непредусмотренное поле (additionalProperties=false)")
        for k, sub in props.items():
            if k in value:
                _validate(sub, value[k], f"{path}.{k}", errors)
    elif typ == "array":
        item_schema = schema.get("items")
        if item_schema:
            for i, item in enumerate(value):
                _validate(item_schema, item, f"{path}[{i}]", errors)
