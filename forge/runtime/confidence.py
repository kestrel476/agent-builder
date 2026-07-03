"""Сопоставление проверенного вывода со статусом: ok / partial / low_confidence / error.

Техническое задание требует явно различать успех, частичный результат, низкую
уверенность и ошибку. Именно здесь живёт эта политика, чтобы каждый сгенерированный
агент сообщал статус одинаково.
"""

from __future__ import annotations

from typing import Any


def assess(
    output: dict[str, Any],
    *,
    threshold: float,
    validation_errors: list[str],
) -> tuple[str, float | None, list[str]]:
    """Вернуть кортеж (статус, уверенность, предупреждения)."""
    warnings: list[str] = []
    confidence = output.get("confidence") if isinstance(output, dict) else None

    required_missing = [e for e in validation_errors if "обязательное поле отсутствует" in e]
    hard_errors = [e for e in validation_errors if e not in required_missing]

    if hard_errors:
        # Жёсткие ошибки несёт список errors в _finish_structured — не дублируем их в warnings.
        return "error", confidence, []

    if required_missing:
        warnings.extend(required_missing)
        status = "partial"
    else:
        status = "ok"

    if isinstance(confidence, (int, float)) and confidence < threshold:
        warnings.append(f"уверенность {confidence:.2f} ниже порога {threshold:.2f}")
        # Низкая уверенность никогда не повышает частичный результат, лишь помечает чистый.
        if status == "ok":
            status = "low_confidence"

    return status, confidence, warnings
