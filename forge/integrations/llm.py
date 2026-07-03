"""Доступ к LLM для Forge — переиспользование провайдера GigaChat из devassist.

Цели проектирования:

* **Переиспользуй, а не переписывай.** Когда учётные данные присутствуют, мы
  управляем реальным ``devassist.llm.gigachat.GigaChatProvider`` (OAuth/mTLS,
  повторы, вызов функций) через тонкий адаптер.
* **Всегда запускаемо.** Без учётных данных детерминированный бэкенд-*заглушка*
  поддерживает работу всего конвейера (и сгенерированных агентов) офлайн и
  воспроизводимо — именно это и использует набор тестов.
* **Один узкий интерфейс.** Всё, что выше этого модуля, видит только
  :class:`ForgeLLM` с ``.text()`` и ``.structured()``. Структурированные вызовы
  идут через *вызов функций* GigaChat (единственный принудительный инструмент,
  параметрами которого является запрошенная JSON-схема), поэтому мы получаем
  валидированные объекты, а не разобранную прозу.
"""

from __future__ import annotations

import json
import re
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from ._vendor import load_devassist_llm
from .stub_brain import StubBrain


class LLMUnavailable(RuntimeError):
    """Вызывается, когда был запрошен реальный бэкенд LLM, но его не удалось построить."""


@dataclass
class CallRecord:
    """Один вызов LLM, сохраняемый для наблюдаемости (`forge status`)."""

    name: str
    backend: str
    kind: str  # "text" | "structured"
    ok: bool
    tokens: int = 0
    note: str = ""


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
@dataclass
class _Response:
    text: str = ""
    data: dict[str, Any] | None = None
    tokens: int = 0


class _Backend(ABC):
    name: str

    @abstractmethod
    def chat(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any] | None,
        context: dict[str, Any] | None,
        task: str,
        temperature: float,
    ) -> _Response: ...


class GigaChatBackend(_Backend):
    """Адаптер над провайдером GigaChat из devassist."""

    name = "gigachat"

    def __init__(self, model: str | None = None) -> None:
        llm = load_devassist_llm()
        if llm is None:  # pragma: no cover - depends on environment
            raise LLMUnavailable(
                "Не удалось импортировать 'devassist.llm'. Укажите DEVASSIST_PATH, "
                "указывающий на репозиторий devassist, или запустите с --offline."
            )
        # Импортируем лениво разрешаемые символы из переиспользуемого пакета.
        from devassist.config import Config
        from devassist.llm.gigachat import GigaChatProvider
        from devassist.llm.types import Message, ToolSpec

        self._Message = Message
        self._ToolSpec = ToolSpec
        cfg = Config.load(model=model)
        try:
            cfg.require_credentials()
        except RuntimeError as e:
            raise LLMUnavailable(str(e)) from e
        self._provider = GigaChatProvider(cfg)

    @property
    def model(self) -> str:
        return self._provider.model

    def chat(self, *, system, user, schema, context, task, temperature) -> _Response:
        Message = self._Message
        messages = [Message(role="system", content=system), Message(role="user", content=user)]
        tools = None
        if schema is not None:
            tool_name = "emit_result"
            tools = [
                self._ToolSpec(
                    name=tool_name,
                    description=(
                        "Верните итоговый результат. Вы ОБЯЗАНЫ вызвать эту функцию "
                        "с аргументами, удовлетворяющими схеме; не отвечайте прозой."
                    ),
                    parameters=schema,
                )
            ]
        turn = self._provider.complete(messages, tools, temperature=temperature)
        tokens = int(turn.usage.get("total_tokens", 0) or 0)
        if schema is not None:
            fc = turn.message.function_call
            if fc is not None:
                return _Response(data=dict(fc.arguments or {}), tokens=tokens)
            content = turn.message.content or ""
            # GigaChat нередко «забывает» вызвать функцию и пишет результат прозой,
            # строгим JSON или python-вызовом emit_result(key=[...]). Терпимый разбор
            # покрывает все три; реформат — крайнее средство.
            data = _parse_loose(content, schema)
            if data is None and content.strip():
                data = self._reformat(content, schema)
            return _Response(text=content, data=data, tokens=tokens)
        return _Response(text=turn.message.content or "", tokens=tokens)

    def _reformat(self, content: str, schema: dict[str, Any]) -> dict[str, Any] | None:
        Message = self._Message
        msgs = [
            Message(role="system", content="Ты конвертер в JSON. Верни ТОЛЬКО валидный JSON по схеме, без markdown и пояснений."),
            Message(role="user", content=f"СХЕМА: {json.dumps(schema, ensure_ascii=False)}\n\nПреобразуй в JSON по схеме этот ответ:\n{content[:8000]}"),
        ]
        try:
            turn = self._provider.complete(msgs, None, temperature=0.0)
        except Exception:
            return None
        return _extract_json(turn.message.content or "")

    def embed(self, texts: list[str], *, batch: int = 64) -> list[list[float]]:
        """Эмбеддинги через тот же httpx-клиент и авторизацию провайдера."""
        p = self._provider
        url = f"{p._cfg.base_url}/embeddings"
        out: list[list[float]] = []
        for i in range(0, len(texts), batch):
            chunk = texts[i : i + batch]
            resp = p._request_with_retry(
                lambda: p._client.post(
                    url,
                    json={"model": "Embeddings", "input": chunk},
                    headers={"Content-Type": "application/json", **p._auth_headers()},
                )
            )
            resp.raise_for_status()
            out.extend(d["embedding"] for d in resp.json()["data"])
        return out


class StubBackend(_Backend):
    """Детерминированный офлайн-движок (заглушка). Воспроизводимый, без сети."""

    name = "stub"

    def __init__(self) -> None:
        self._brain = StubBrain()

    def chat(self, *, system, user, schema, context, task, temperature) -> _Response:
        if schema is not None:
            data = self._brain.structured(task=task, context=context or {}, schema=schema, user=user)
            return _Response(data=data, tokens=0)
        text = self._brain.text(task=task, context=context or {}, user=user)
        return _Response(text=text, tokens=0)

    def embed(self, texts: list[str], **_) -> list[list[float]]:
        """Детерминированные «эмбеддинги» из хешированных триграмм символов.

        Не семантика, но устойчивая лексическая близость — достаточно, чтобы RAG
        работал офлайн воспроизводимо (в тестах/CI)."""
        import math
        import zlib

        dim = 256
        out: list[list[float]] = []
        for t in texts:
            vec = [0.0] * dim
            s = "  " + (t or "").lower() + "  "
            for i in range(len(s) - 2):
                h = zlib.crc32(s[i : i + 3].encode("utf-8")) % dim  # стабильный хеш
                vec[h] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


# --------------------------------------------------------------------------- #
# Facade
# --------------------------------------------------------------------------- #
class ForgeLLM:
    """Не зависящий от провайдера фасад LLM, используемый везде в Forge.

    ``structured`` возвращает обычный ``dict``, валидированный по ``schema`` (с
    одним автоматическим циклом починки на реальном бэкенде). ``context`` несёт
    машиночитаемые входные данные, нужные офлайн-движку (заглушке); реальный
    бэкенд его игнорирует (всё, что ему нужно, уже включено в ``user``).
    """

    def __init__(self, backend: _Backend, *, offline: bool) -> None:
        self._backend = backend
        self.offline = offline
        self.calls: list[CallRecord] = []
        self._calls_lock = threading.Lock()

    def _record(self, rec: CallRecord) -> None:
        with self._calls_lock:  # вызовы могут идти из нескольких потоков
            self.calls.append(rec)

    @property
    def backend_name(self) -> str:
        return self._backend.name

    @property
    def model(self) -> str:
        return getattr(self._backend, "model", "stub")

    @property
    def total_tokens(self) -> int:
        return sum(c.tokens for c in self.calls)

    # -- embeddings ------------------------------------------------------- #
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Векторные эмбеддинги для RAG. Реальный бэкенд — GigaChat `Embeddings`,
        офлайн — детерминированные хеш-векторы."""
        if not texts:
            return []
        vecs = self._backend.embed(texts)
        self._record(CallRecord(name="embed", backend=self._backend.name, kind="embed", ok=bool(vecs), tokens=0, note=f"{len(texts)} текст(ов)"))
        return vecs

    # -- text ------------------------------------------------------------- #
    def text(self, *, task: str, system: str, user: str,
             context: dict[str, Any] | None = None, temperature: float = 0.2) -> str:
        resp = self._backend.chat(
            system=system, user=user, schema=None, context=context, task=task, temperature=temperature
        )
        self._record(CallRecord(name=task, backend=self._backend.name, kind="text", ok=bool(resp.text), tokens=resp.tokens))
        return resp.text

    # -- structured ------------------------------------------------------- #
    def structured(
        self,
        *,
        task: str,
        system: str,
        user: str,
        schema: dict[str, Any],
        context: dict[str, Any] | None = None,
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        resp = self._backend.chat(
            system=system, user=user, schema=schema, context=context, task=task, temperature=temperature
        )
        data = resp.data
        ok = isinstance(data, dict)
        missing = _schema_missing(schema, data) if ok else ["<объект не возвращён>"]
        if missing and not self.offline:
            # Одна попытка починки: сообщаем модели, какие именно поля неверны.
            repair = (
                f"{user}\n\n[VALIDATION] Ваш предыдущий ответ содержал пропущенные "
                f"или некорректные поля: {', '.join(missing)}. Верните полный объект снова."
            )
            resp = self._backend.chat(
                system=system, user=repair, schema=schema, context=context, task=task, temperature=0.0
            )
            data = resp.data
            ok = isinstance(data, dict)
            missing = _schema_missing(schema, data) if ok else ["<объект не возвращён>"]

        self._record(
            CallRecord(
                name=task,
                backend=self._backend.name,
                kind="structured",
                ok=ok and not missing,
                tokens=resp.tokens,
                note="" if not missing else f"пропущено: {', '.join(missing)}",
            )
        )
        if not ok:
            raise LLMUnavailable(f"Структурированный вызов '{task}' не вернул объект.")
        return data


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #
def build_llm(*, offline: bool, model: str | None = None) -> ForgeLLM:
    """Строит :class:`ForgeLLM`.

    ``offline=True`` принудительно использует детерминированную заглушку. Иначе мы
    пробуем GigaChat и переходим на заглушку только если её не удаётся построить
    (нет учётных данных / отсутствует репозиторий).
    """
    if offline:
        return ForgeLLM(StubBackend(), offline=True)
    try:
        return ForgeLLM(GigaChatBackend(model=model), offline=False)
    except LLMUnavailable:
        return ForgeLLM(StubBackend(), offline=True)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _strip_reasoning(text: str) -> str:
    """Убирает блоки рассуждений ``<think>...</think>`` (reasoning-модели GigaChat)."""
    if not text:
        return text
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I).strip()


def _extract_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    text = _strip_reasoning(text).strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[-1] if "\n" in text else text
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _balanced(s: str) -> str | None:
    """Сбалансированный фрагмент [...] или {...} от начала строки (учёт строк/экранов)."""
    if not s or s[0] not in "[{":
        return None
    open_c = s[0]
    close_c = "]" if open_c == "[" else "}"
    depth = 0
    instr: str | None = None
    for i, ch in enumerate(s):
        if instr:
            if ch == instr and s[i - 1] != "\\":
                instr = None
        elif ch in "\"'":
            instr = ch
        elif ch == open_c:
            depth += 1
        elif ch == close_c:
            depth -= 1
            if depth == 0:
                return s[: i + 1]
    return None


def _parse_loose(content: str, schema: dict[str, Any]) -> dict[str, Any] | None:
    """Терпимый разбор ответа модели в объект по схеме.

    Покрывает три формы, которые GigaChat выдаёт, когда «забывает» про
    function calling: строгий JSON ``{...}``; вызов в python-синтаксисе
    ``emit_result(situations=[...])``; и просто значения ``key: [...]``.
    Значения парсятся как JSON или python-литерал (одинарные кавычки).
    """
    import ast

    content = _strip_reasoning(content)
    j = _extract_json(content)
    if isinstance(j, dict):
        return j
    out: dict[str, Any] = {}
    for key in schema.get("properties", {}):
        m = re.search(rf'["\']?{re.escape(key)}["\']?\s*[:=]\s*', content)
        if not m:
            continue
        rest = content[m.end():].lstrip()
        if not rest or rest[0] not in "[{":
            continue
        frag = _balanced(rest)
        if not frag:
            continue
        for parser in (json.loads, ast.literal_eval):
            try:
                out[key] = parser(frag)
                break
            except Exception:
                continue
    return out or None


def _schema_missing(schema: dict[str, Any], data: dict[str, Any] | None) -> list[str]:
    """Лёгкая проверка обязательных полей (полная валидация выполняется в рантайме)."""
    if not isinstance(data, dict):
        return ["<не объект>"]
    required = schema.get("required") or []
    return [f for f in required if f not in data or data[f] in (None, "")]
