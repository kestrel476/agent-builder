"""Извлечение текста из документов для Forge — переиспользование сервиса ``extractors``.

Переиспользуемый пакет обрабатывает более 40 форматов (PDF/DOCX/XLSX/EML/HTML/изображения→OCR, …)
за один вызов. Мы оборачиваем его так, чтобы Forge и генерируемые им агенты зависели
только от :class:`DocumentReader`. Если пакет недоступен, мы деградируем до простого
чтения текста в UTF-8, чтобы конвейер продолжал работать для входных файлов ``.txt``/``.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._vendor import load_extractors


@dataclass
class ReadResult:
    """Нормализованный результат чтения одного документа."""

    text: str
    status: str  # "ok" | "no_text_layer" | "unsupported" | "error"
    needs_ocr: bool = False
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    backend: str = "extractors"

    @property
    def ok(self) -> bool:
        return self.status == "ok" and bool(self.text)


class DocumentReader:
    """Тонкий фасад над переиспользуемым пакетом ``extractors`` (ленивый, кэширующий)."""

    def __init__(self, *, pdf_max_pages: int | None = None) -> None:
        self._pdf_max_pages = pdf_max_pages
        self._service: Any = None
        self._FileSource: Any = None
        self._available: bool | None = None

    def _ensure(self) -> bool:
        if self._available is not None:
            return self._available
        pkg = load_extractors()
        if pkg is None:
            self._available = False
            return False
        try:
            self._service = pkg.build_default_extractor(pdf_max_pages=self._pdf_max_pages)
            self._FileSource = pkg.FileSource
            self._available = True
        except Exception:  # pragma: no cover - missing optional parsers
            self._available = False
        return self._available

    @property
    def available(self) -> bool:
        return self._ensure()

    def read(self, path: str | Path) -> ReadResult:
        path = Path(path)
        if not path.exists():
            return ReadResult(text="", status="error", error=f"Файл не найден: {path}")
        if self._ensure():
            res = self._service.extract(self._FileSource(path=str(path)))
            return ReadResult(
                text=res.text or "",
                status=res.status.value,
                needs_ocr=bool(res.needs_ocr),
                error=res.error,
                meta=dict(res.meta or {}),
            )
        return self._fallback_read(path)

    @staticmethod
    def _fallback_read(path: Path) -> ReadResult:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            return ReadResult(text=text, status="ok" if text else "error", backend="plaintext-fallback")
        except Exception as e:  # pragma: no cover
            return ReadResult(text="", status="error", error=str(e), backend="plaintext-fallback")
