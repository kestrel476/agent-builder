"""Чертёж (внутреннее представление, IR) и все его подструктуры.

Чертёж намеренно *явный*: допущения, открытые замечания, неразрешённые
неоднозначности и непроверенное поведение — это полноценные поля, а не скрытое
состояние. В этом и состоит весь смысл Forge — выявлять неопределённость, а не
молча угадывать.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Перечисления
# --------------------------------------------------------------------------- #
class Stage(str, Enum):
    """Этап жизненного цикла чертежа."""

    DRAFT = "draft"            # создан, ещё ничего не проанализировано
    INTAKE = "intake"          # сырые инструкции структурированы
    CLARIFYING = "clarifying"  # есть открытые вопросы
    CONTRACTED = "contracted"  # контракт вход/выход зафиксирован
    BUILT = "built"            # бандл агента синтезирован
    TESTED = "tested"          # набор тестов запущен хотя бы раз
    PACKAGED = "packaged"      # итоговый бандл выпущен


class Severity(str, Enum):
    BLOCKER = "blocker"  # нельзя ответственно генерировать, пока не разрешено
    MAJOR = "major"      # вероятно приведёт к неверному поведению
    MINOR = "minor"      # стоит разрешить, но не блокирует
    INFO = "info"


class IssueKind(str, Enum):
    AMBIGUITY = "ambiguity"
    CONTRADICTION = "contradiction"
    MISSING = "missing"
    VAGUE_SUCCESS = "vague_success"
    EXAMPLE_CONFLICT = "example_conflict"
    RISK = "risk"
    SCOPE = "scope"


class IssueStatus(str, Enum):
    OPEN = "open"
    RESOLVED = "resolved"    # дан ответ пользователем
    ASSUMED = "assumed"      # авторазрешено через запись допущения
    WAIVED = "waived"        # пользователь явно отложил


class Capability(str, Enum):
    LLM = "llm"
    RAG = "rag"
    OCR = "ocr"
    DOC_EXTRACTION = "doc_extraction"
    JSON_OUTPUT = "json_output"
    SCHEMA_VALIDATION = "schema_validation"
    ROUTING = "routing"
    RULES = "rules"
    CONFIDENCE = "confidence"
    LOGGING = "logging"
    MULTI_STEP = "multi_step"
    DRY_RUN = "dry_run"


class OutputKind(str, Enum):
    JSON = "json"
    CLASSIFICATION = "classification"
    EDITED_DOCUMENT = "edited_document"
    NEW_DOCUMENT = "new_document"
    CHAT = "chat"
    REPORT = "report"


# --------------------------------------------------------------------------- #
# Структурированные инструкции этапа 0
# --------------------------------------------------------------------------- #
class StructuredInstructions(BaseModel):
    """Нормализованная форма свободного текста юридических инструкций (выход этапа 0)."""

    goal: str = ""
    scope_in: list[str] = Field(default_factory=list)
    scope_out: list[str] = Field(default_factory=list)
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    business_rules: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    definitions: list[str] = Field(default_factory=list)
    ambiguities: list[str] = Field(default_factory=list)
    missing_data: list[str] = Field(default_factory=list)
    interpretation_risks: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Контракт вход/выход
# --------------------------------------------------------------------------- #
class InputSpec(BaseModel):
    name: str
    kind: str = "text"          # text | json | document | file
    required: bool = True
    formats: list[str] = Field(default_factory=list)  # напр. ["pdf", "docx"]
    description: str = ""


class FieldSpec(BaseModel):
    """Одно поле выхода; преобразуется в JSON-схему в контракте."""

    name: str
    type: str = "string"        # string | number | integer | boolean | array | object
    required: bool = False
    description: str = ""
    enum: list[str] = Field(default_factory=list)
    # Для type="array" из объектов: схема одного элемента (иначе элементы — строки).
    item_fields: list["FieldSpec"] = Field(default_factory=list)
    example: Any = None

    def to_schema(self) -> dict[str, Any]:
        node: dict[str, Any] = {"type": self.type}
        if self.description:
            node["description"] = self.description
        if self.enum:
            node["enum"] = list(self.enum)
        if self.type == "array":
            if self.item_fields:
                node["items"] = {
                    "type": "object",
                    "properties": {sf.name: sf.to_schema() for sf in self.item_fields},
                    "required": [sf.name for sf in self.item_fields if sf.required],
                    "additionalProperties": False,
                }
            else:
                node["items"] = {"type": "string"}
        return node


class IOContract(BaseModel):
    """Явный, тестируемый контракт вход/выход для агента."""

    inputs: list[InputSpec] = Field(default_factory=list)
    output_kind: OutputKind = OutputKind.JSON
    fields: list[FieldSpec] = Field(default_factory=list)  # для JSON-подобных выходов
    success_criteria: list[str] = Field(default_factory=list)
    error_definition: str = ""
    partial_result_policy: str = ""
    confidence_required: bool = True

    def output_schema(self) -> dict[str, Any]:
        """JSON-схема для выхода агента (всегда объект-обёртка)."""
        props: dict[str, Any] = {f.name: f.to_schema() for f in self.fields}
        required = [f.name for f in self.fields if f.required]
        if self.confidence_required:
            props["confidence"] = {"type": "number", "description": "Самооценка уверенности от 0 до 1."}
            # Обязательность заставляет модель (через function calling) всегда
            # возвращать уверенность — иначе она часто её опускает.
            required.append("confidence")
        return {
            "type": "object",
            "properties": props,
            "required": required,
            "additionalProperties": False,
        }


# --------------------------------------------------------------------------- #
# Замечания / вопросы / допущения
# --------------------------------------------------------------------------- #
class Issue(BaseModel):
    id: str
    kind: IssueKind
    severity: Severity = Severity.MAJOR
    message: str
    where: str = ""             # путь к полю чертежа, которого касается замечание
    rationale: str = ""         # почему это важно
    suggestion: str = ""        # предлагаемое разрешение / значение по умолчанию
    source: str = ""            # какой критик его поднял
    status: IssueStatus = IssueStatus.OPEN

    @property
    def blocking(self) -> bool:
        return self.severity == Severity.BLOCKER and self.status == IssueStatus.OPEN


class Question(BaseModel):
    id: str
    text: str
    why: str = ""
    category: str = "general"
    options: list[str] = Field(default_factory=list)
    blocking: bool = False
    issue_id: str | None = None
    answer: str | None = None

    @property
    def answered(self) -> bool:
        return self.answer is not None and self.answer.strip() != ""


class Assumption(BaseModel):
    """Явно зафиксированное значение по умолчанию, принятое Forge вместо ответа."""

    id: str
    statement: str
    because: str = ""
    risk: Severity = Severity.MINOR
    issue_id: str | None = None
    confirmed: bool | None = None  # None = не подтверждено, True/False после проверки


# --------------------------------------------------------------------------- #
# Тесты
# --------------------------------------------------------------------------- #
class TestCase(BaseModel):
    __test__ = False  # это доменная модель, а не тестовый класс pytest

    id: str
    name: str
    # При запуске используется ровно один источник входа, в таком порядке приоритета:
    input_text: str | None = None
    input_json: dict[str, Any] | None = None
    input_file: str | None = None      # путь относительно рабочей папки
    input_files: list[str] = Field(default_factory=list)  # несколько документов (compare/package)
    expected: dict[str, Any] | None = None  # ожидаемый объект выхода (совпадение по подмножеству)
    expected_text: str | None = None        # для выходов документ/чат
    must_contain: list[str] = Field(default_factory=list)
    # Метрика precision/recall для list-полей (агенты детекции): точное совпадение
    # неуместно — важна полнота. Напр. {"field":"detected_risks","key":"risk_code",
    # "expected":["ObvR-1"],"min_recall":1.0,"min_precision":0.0}.
    metric: dict[str, Any] | None = None
    tags: list[str] = Field(default_factory=list)


class HistoryEvent(BaseModel):
    seq: int
    stage: Stage
    action: str
    detail: str = ""


# --------------------------------------------------------------------------- #
# Чертёж
# --------------------------------------------------------------------------- #
class AgentBlueprint(BaseModel):
    """Полная спецификация одного агента в процессе создания."""

    name: str
    slug: str
    archetype: str = "json_extraction"
    version: int = 1
    stage: Stage = Stage.DRAFT

    goal: str = ""
    domain_notes: str = ""
    glossary: dict[str, str] = Field(default_factory=dict)

    instructions_raw: str = ""
    instructions: StructuredInstructions = Field(default_factory=StructuredInstructions)

    io: IOContract = Field(default_factory=IOContract)
    capabilities: list[Capability] = Field(default_factory=list)
    # Конфигурация внешней базы знаний для RAG-архетипов (источник каталога +
    # маппинг полей + фильтр). См. forge.runtime.knowledge.KnowledgeConfig.
    knowledge: dict[str, Any] | None = None
    # Каталог правил для rule_check (список правил-данных). См. forge.runtime.rules.
    rule_catalog: list[dict[str, Any]] | None = None
    # Конфигурация маршрутизации пакета: {taxonomy, required, per_type}. См. package_review.
    package: dict[str, Any] | None = None
    # Каталог проверок «да/нет» для checklist-экспертизы. См. checklist_review.
    checklist: list[dict[str, Any]] | None = None

    issues: list[Issue] = Field(default_factory=list)
    questions: list[Question] = Field(default_factory=list)
    assumptions: list[Assumption] = Field(default_factory=list)
    test_cases: list[TestCase] = Field(default_factory=list)

    # Синтезированные артефакты (хранятся в чертеже для воспроизводимости).
    system_prompt: str = ""
    prompt_addenda: list[str] = Field(default_factory=list)

    history: list[HistoryEvent] = Field(default_factory=list)

    # -- удобные представления ------------------------------------------- #
    @property
    def open_issues(self) -> list[Issue]:
        return [i for i in self.issues if i.status == IssueStatus.OPEN]

    @property
    def blocking_issues(self) -> list[Issue]:
        return [i for i in self.issues if i.blocking]

    @property
    def open_questions(self) -> list[Question]:
        return [q for q in self.questions if not q.answered]

    def has_capability(self, cap: Capability) -> bool:
        return cap in self.capabilities

    def effective_prompt(self) -> str:
        parts = [self.system_prompt] + [a for a in self.prompt_addenda if a.strip()]
        return "\n\n".join(p for p in parts if p.strip())

    # -- помощники мутации ----------------------------------------------- #
    def next_id(self, prefix: str) -> str:
        existing = {
            "ISS": len(self.issues),
            "Q": len(self.questions),
            "ASM": len(self.assumptions),
            "T": len(self.test_cases),
        }.get(prefix, 0)
        return f"{prefix}-{existing + 1:03d}"

    def record(self, action: str, detail: str = "") -> None:
        self.history.append(
            HistoryEvent(seq=len(self.history) + 1, stage=self.stage, action=action, detail=detail)
        )

    def add_issue(self, issue: Issue) -> Issue:
        # Дедупликация по (kind, where, message), чтобы избежать спама критиков между прогонами.
        key = (issue.kind, issue.where, issue.message)
        for existing in self.issues:
            if (existing.kind, existing.where, existing.message) == key:
                return existing
        self.issues.append(issue)
        return issue

    def resolve_issue(self, issue_id: str, status: IssueStatus = IssueStatus.RESOLVED) -> None:
        for i in self.issues:
            if i.id == issue_id:
                i.status = status
