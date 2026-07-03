"""База знаний для RAG-агентов: каталог записей + эмбеддинг-индекс + retrieval.

Каталог — это JSON: либо словарь ``{код: {поля...}}``, либо список объектов.
Конфигурация (из ``agent.yaml``) задаёт, как из записи взять идентификатор,
заголовок и «факт», и опциональный фильтр (например, только определённой
категории риска). Индекс эмбеддингов строится один раз и кэшируется рядом с
каталогом, чтобы повторные запуски были быстрыми.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class KnowledgeConfig:
    file: str                       # путь к каталогу относительно бандла
    id_field: str = "__key__"       # "__key__" = ключ словаря; иначе имя поля
    title_field: str = "title"
    fact_field: str = "fact"
    filter: dict[str, Any] = field(default_factory=dict)  # поле -> требуемое значение
    top_k: int = 30

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "KnowledgeConfig":
        known = {f: d[f] for f in cls.__dataclass_fields__ if f in d}
        return cls(**known)


@dataclass
class Entry:
    code: str
    title: str
    fact: str


class KnowledgeBase:
    """Загруженный каталог + (лениво построенный) эмбеддинг-индекс."""

    def __init__(self, entries: list[Entry], *, bundle_dir: Path, cfg: KnowledgeConfig) -> None:
        self.entries = entries
        self.bundle_dir = Path(bundle_dir)
        self.cfg = cfg
        self._matrix = None  # numpy [n, dim], нормированная
        self._codes = [e.code for e in entries]
        self._by_code = {e.code: e for e in entries}  # O(1) доступ (горячий путь RAG)

    # -- загрузка --------------------------------------------------------- #
    @classmethod
    def load(cls, bundle_dir: str | Path, cfg: KnowledgeConfig) -> "KnowledgeBase":
        bundle_dir = Path(bundle_dir)
        raw = json.loads((bundle_dir / cfg.file).read_text(encoding="utf-8"))
        items = raw.items() if isinstance(raw, dict) else enumerate(raw)
        entries: list[Entry] = []
        for key, obj in items:
            if not isinstance(obj, dict):
                continue
            if cfg.filter and any(obj.get(f) != v for f, v in cfg.filter.items()):
                continue
            code = str(key) if cfg.id_field == "__key__" else str(obj.get(cfg.id_field, key))
            entries.append(Entry(
                code=code,
                title=str(obj.get(cfg.title_field, "")),
                fact=str(obj.get(cfg.fact_field, "") or ""),
            ))
        return cls(entries, bundle_dir=bundle_dir, cfg=cfg)

    # -- индекс ----------------------------------------------------------- #
    def _entry_text(self, e: Entry) -> str:
        fact = re.sub(r"_+", " ", e.fact)  # убрать плейсхолдеры-подчёркивания
        return f"{e.title}. {fact}".strip()

    def _index_path(self, backend: str) -> Path:
        # Имя кэша ПРИВЯЗАНО к бэкенду эмбеддингов: офлайн-стаб (dim 256) и реальный
        # GigaChat (dim 1024) дают разные индексы — нельзя смешивать (иначе при
        # сборке офлайн и запуске онлайн q@matrix.T падает или выдаёт мусор).
        return self.bundle_dir / "knowledge" / f"index-{backend}.npz"

    def ensure_index(self, llm) -> None:
        """Построить или загрузить кэш эмбеддингов записей каталога (по бэкенду)."""
        import numpy as np

        backend = getattr(llm, "backend_name", "unknown")
        path = self._index_path(backend)
        if path.is_file():
            z = np.load(path, allow_pickle=True)
            if list(z["codes"]) == self._codes:
                self._matrix = self._unit(z["emb"].astype("float32"))
                return
        vecs = llm.embed([self._entry_text(e) for e in self.entries])
        emb = np.asarray(vecs, dtype="float32")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, emb=emb, codes=np.asarray(self._codes, dtype=object))
        self._matrix = self._unit(emb)

    def retrieve(self, query_vecs: list[list[float]], k: int | None = None) -> list[str]:
        """Объединение топ-K кодов по косинусной близости для каждого запроса."""
        import numpy as np

        assert self._matrix is not None, "сначала вызовите ensure_index()"
        k = k or self.cfg.top_k
        q = self._unit(np.asarray(query_vecs, dtype="float32"))
        if q.shape[1] != self._matrix.shape[1]:  # подстраховка от рассинхрона размерностей
            raise ValueError(
                f"Размерность эмбеддингов запроса ({q.shape[1]}) ≠ индекса ({self._matrix.shape[1]}). "
                "Индекс собран другим бэкендом — удалите knowledge/index-*.npz и пересоберите."
            )
        sims = q @ self._matrix.T                       # [запросы, записи]
        out: list[str] = []
        seen: set[str] = set()
        for row in sims:
            for idx in np.argsort(-row)[:k]:
                c = self._codes[idx]
                if c not in seen:
                    seen.add(c)
                    out.append(c)
        return out

    def get(self, code: str) -> Entry | None:
        return self._by_code.get(code)

    @staticmethod
    def _unit(m):
        import numpy as np

        return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)
