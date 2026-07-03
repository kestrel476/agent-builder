"""Встроенные архетипы для типичных агентов автоматизации юридических задач.

Каждый объявляет значения по умолчанию и небольшой набор *блокирующих*
уточняющих вопросов — тех, которые, как показывает опыт, чаще всего остаются
неуказанными для агента такого вида.
"""

from __future__ import annotations

from ..model import Capability, FieldSpec, OutputKind
from .base import AgentArchetype, register

_C = Capability

register(AgentArchetype(
    key="json_extraction",
    title="Документ → извлечение в JSON",
    description="Извлечение фиксированного набора полей из юридических документов в структурированный JSON.",
    runtime_strategy="extract",
    output_kind=OutputKind.JSON,
    capabilities=(_C.LLM, _C.DOC_EXTRACTION, _C.OCR, _C.JSON_OUTPUT, _C.SCHEMA_VALIDATION,
                  _C.CONFIDENCE, _C.LOGGING),
    # Без зашитых полей: схема извлечения целиком зависит от инструкции и
    # выводится на этапе контракта (LLM/офлайн-движком), а не навязывается архетипом.
    default_fields=(),
    clarifiers=(
        ("Какие именно поля должны быть извлечены, и какие из них обязательные, а какие — необязательные?",
         "Схема выхода — это контракт; размытые списки полей делают тесты невозможными.", "io", True),
        ("Когда поле отсутствует в документе, агент должен выдать null, опустить его или завершиться ошибкой?",
         "Определяет разницу между частичным результатом и ошибкой.", "io", True),
        ("Какая нормализация требуется для дат/сумм (ISO-8601? код валюты? дословно как есть)?",
         "Определяет, считаются ли два «правильных» ответа равными в тестах.", "rules", True),
    ),
))

register(AgentArchetype(
    key="classification",
    title="Классификация документов",
    description="Отнесение каждого документа к одной из фиксированного набора категорий.",
    runtime_strategy="extract",
    output_kind=OutputKind.CLASSIFICATION,
    capabilities=(_C.LLM, _C.DOC_EXTRACTION, _C.OCR, _C.JSON_OUTPUT, _C.CONFIDENCE,
                  _C.ROUTING, _C.LOGGING),
    default_fields=(
        FieldSpec(name="label", type="string", required=True, description="Выбранная категория."),
        FieldSpec(name="rationale", type="string", required=False, description="Почему именно эта метка."),
    ),
    clarifiers=(
        ("Каков закрытый список категорий с однострочным определением каждой?",
         "Классификацию можно тестировать только против исчерпывающего, непересекающегося набора меток.", "io", True),
        ("Допускается ли несколько меток, или ровно одна метка на документ?",
         "Меняет как схему, так и критерий успеха.", "io", True),
        ("Что должно происходить ниже порога уверенности — «неизвестно», проверка человеком или наилучшая догадка?",
         "Определяет поведение при низкой уверенности.", "rules", True),
    ),
))

register(AgentArchetype(
    key="risk_check",
    title="Проверка рисков / соответствия требованиям",
    description="Оценка условий и выявление рисков или несоответствующих пунктов.",
    runtime_strategy="extract",
    output_kind=OutputKind.JSON,
    capabilities=(_C.LLM, _C.DOC_EXTRACTION, _C.OCR, _C.RULES, _C.JSON_OUTPUT,
                  _C.CONFIDENCE, _C.LOGGING, _C.MULTI_STEP),
    default_fields=(
        FieldSpec(name="findings", type="array", required=True,
                  description="По одной записи на каждое проверенное условие.",
                  item_fields=[
                      FieldSpec(name="condition", type="string", required=True,
                                description="Название проверяемого условия/риска."),
                      FieldSpec(name="present", type="boolean", required=True,
                                description="Обнаружено ли условие в документе."),
                      FieldSpec(name="quote", type="string", required=False,
                                description="Цитата из документа, подтверждающая вывод."),
                  ]),
        FieldSpec(name="overall_risk", type="string", required=True, enum=["low", "medium", "high"],
                  description="Агрегированный уровень риска."),
    ),
    clarifiers=(
        ("Перечислите каждое правило/условие для проверки и точную формулировку, которая считается нарушением.",
         "Правила рисков должны быть явными и проверяемыми по отдельности.", "rules", True),
        ("Как замечания агрегируются в общий уровень риска?",
         "Иначе поле overall_risk будет недетерминированным.", "rules", True),
        ("Должен ли агент ссылаться на исходный пункт (цитата + расположение) для каждого замечания?",
         "Возможность аудита обычно обязательна при юридической оценке рисков.", "io", False),
    ),
))

register(AgentArchetype(
    key="catalog_risk_detection",
    title="Детекция рисков по каталогу (RAG)",
    description="Поиск в документе рисков из большого внешнего каталога через эмбеддинги и LLM-сопоставление.",
    runtime_strategy="rag_detect",
    output_kind=OutputKind.JSON,
    capabilities=(_C.LLM, _C.RAG, _C.DOC_EXTRACTION, _C.OCR, _C.JSON_OUTPUT, _C.SCHEMA_VALIDATION,
                  _C.CONFIDENCE, _C.MULTI_STEP, _C.LOGGING),
    default_fields=(
        FieldSpec(name="detected_risks", type="array", required=True,
                  description="Обнаруженные в документе риски из каталога.",
                  item_fields=[
                      FieldSpec(name="risk_code", type="string", required=True,
                                description="Код риска из каталога."),
                      FieldSpec(name="title", type="string", required=True,
                                description="Название риска (дословно из каталога)."),
                      FieldSpec(name="fact", type="string", required=True,
                                description="Конкретное обстоятельство из документа, порождающее риск."),
                  ]),
    ),
    clarifiers=(
        ("Каков источник каталога рисков (файл) и какие его поля задают код, название и типовой факт?",
         "Каталог — внешняя база знаний; без сопоставления полей агент не построить.", "io", True),
        ("Нужно ли ограничить поиск подмножеством каталога (например, только определённой категорией риска)?",
         "Сужение пространства поиска резко повышает точность.", "rules", True),
        ("Сколько кандидатов извлекать на одно обстоятельство (top-K) и каков критерий присутствия риска?",
         "Определяет полноту и точность детекции.", "rules", False),
    ),
))

register(AgentArchetype(
    key="checklist",
    title="Экспертиза по чек-листу",
    description="Правовая экспертиза документа по каталогу проверок «да/нет» (аналог экспертиз "
                "guarantee/corporate_approval): LLM решает выполнение условия, вывод берётся шаблоном.",
    runtime_strategy="checklist_review",
    output_kind=OutputKind.REPORT,
    confidence=False,
    capabilities=(_C.LLM, _C.DOC_EXTRACTION, _C.OCR, _C.RULES, _C.JSON_OUTPUT, _C.MULTI_STEP, _C.LOGGING),
    default_fields=(
        FieldSpec(name="results", type="array", required=True,
                  description="Результат по каждой проверке чек-листа.",
                  item_fields=[
                      FieldSpec(name="check_id", type="string", required=True, description="Код проверки."),
                      FieldSpec(name="question", type="string", required=False, description="Вопрос проверки."),
                      FieldSpec(name="verdict", type="string", required=True, description="yes / no / n_a."),
                      FieldSpec(name="position", type="string", required=False, description="Позиция ЮП."),
                      FieldSpec(name="finding", type="string", required=False, description="Описание риска/замечания."),
                      FieldSpec(name="recommendation", type="string", required=False, description="Рекомендация."),
                      FieldSpec(name="evidence", type="string", required=False, description="Цитата из документа."),
                  ]),
        FieldSpec(name="flagged", type="array", required=False,
                  description="Коды сработавших проверок (verdict=yes)."),
        FieldSpec(name="overall_position", type="string", required=True,
                  description="Итоговая позиция по документу."),
    ),
    clarifiers=(
        ("Каков каталог проверок: вопрос «да/нет», признак выполнения, и шаблон вывода (позиция/риск/рекомендация) для каждой?",
         "Чек-лист — данные экспертизы; вывод берётся шаблоном при срабатывании проверки.", "io", True),
        ("Как агрегировать в итоговую позицию (есть риски / есть замечания / соответствует)?",
         "Определяет overall_position.", "rules", False),
    ),
))

register(AgentArchetype(
    key="rule_check",
    title="Проверка по правилам (rule-engine)",
    description="Детерминированная проверка документа по декларативному каталогу правил "
                "(ключевые слова / близость / regex), без LLM.",
    runtime_strategy="rule_check",
    output_kind=OutputKind.JSON,
    confidence=False,
    capabilities=(_C.RULES, _C.DOC_EXTRACTION, _C.OCR, _C.JSON_OUTPUT, _C.LOGGING),
    default_fields=(
        FieldSpec(name="matched_rules", type="array", required=True,
                  description="Сработавшие правила.",
                  item_fields=[
                      FieldSpec(name="rule_id", type="string", required=True, description="Код правила."),
                      FieldSpec(name="title", type="string", required=False, description="Название правила."),
                      FieldSpec(name="severity", type="string", required=False, description="Уровень."),
                      FieldSpec(name="evidence", type="string", required=False, description="Фрагмент-подтверждение."),
                  ]),
    ),
    clarifiers=(
        ("Откуда берётся каталог правил и какие поля у правила (id, тип матча, термины/паттерн)?",
         "Правила — данные движка; без них агент не построить.", "io", True),
        ("Какие типы матча нужны: ключевые слова (any/all), близость (near), регулярные выражения?",
         "Определяет выразительность правил.", "rules", False),
    ),
))

register(AgentArchetype(
    key="document_package",
    title="Обработка пакета документов (кейса)",
    description="Приём пакета из множества документов: классификация типа каждого → маршрутизация → "
                "пер-документный анализ → агрегация в единое заключение с проверкой обязательных типов.",
    runtime_strategy="package_review",
    output_kind=OutputKind.REPORT,
    confidence=False,
    input_arity=9,  # много документов
    capabilities=(_C.LLM, _C.DOC_EXTRACTION, _C.OCR, _C.ROUTING, _C.MULTI_STEP, _C.JSON_OUTPUT,
                  _C.SCHEMA_VALIDATION, _C.LOGGING),
    default_fields=(
        FieldSpec(name="verdict", type="string", required=True,
                  enum=["success", "partial", "rejected"], description="Итоговый вердикт по пакету."),
        FieldSpec(name="documents", type="array", required=True,
                  description="Разбор каждого документа пакета.",
                  item_fields=[
                      FieldSpec(name="name", type="string", required=True, description="Имя файла."),
                      FieldSpec(name="doc_type", type="string", required=True, description="Определённый тип."),
                      FieldSpec(name="summary", type="string", required=False, description="Краткое содержание."),
                      FieldSpec(name="findings", type="array", required=False, description="Находки/риски."),
                  ]),
        FieldSpec(name="flows", type="array", required=False,
                  description="Плоский список находок с привязкой к документу.",
                  item_fields=[
                      FieldSpec(name="document", type="string", required=True),
                      FieldSpec(name="doc_type", type="string", required=False),
                      FieldSpec(name="finding", type="string", required=True),
                  ]),
        FieldSpec(name="missing_required", type="array", required=False,
                  description="Обязательные типы документов, которых нет в пакете."),
        FieldSpec(name="unrouted", type="array", required=False,
                  description="Документы нераспознанного типа."),
        FieldSpec(name="case_summary", type="string", required=False, description="Сводное заключение по кейсу."),
    ),
    clarifiers=(
        ("Какова таксономия типов документов (код + описание каждого) для классификации?",
         "По типу маршрутизируется анализ; без таксономии нельзя классифицировать.", "io", True),
        ("Какие типы документов ОБЯЗАТЕЛЬНЫ в пакете (для проверки комплектности)?",
         "Определяет вердикт rejected при неполном пакете.", "rules", True),
        ("Что именно извлекать/проверять по каждому типу документа?",
         "Задаёт пер-документный анализ.", "rules", False),
    ),
))

register(AgentArchetype(
    key="clause_comparison",
    title="Сравнение пунктов / редлайн",
    description="Сравнение положений между двумя документами (например, проект против стандарта).",
    runtime_strategy="compare",
    output_kind=OutputKind.JSON,
    capabilities=(_C.LLM, _C.DOC_EXTRACTION, _C.OCR, _C.JSON_OUTPUT, _C.CONFIDENCE,
                  _C.MULTI_STEP, _C.LOGGING),
    input_arity=2,
    default_fields=(
        FieldSpec(name="differences", type="array", required=True, description="Различия по каждому пункту."),
        FieldSpec(name="missing_in_b", type="array", required=False, description="Пункты, присутствующие в A, но отсутствующие в B."),
    ),
    clarifiers=(
        ("Какой документ является эталоном (стандартом), а какой проверяется?",
         "Направление определяет смысл «добавлено» против «удалено».", "io", True),
        ("Сравнивать пункт за пунктом по заголовкам или семантически по теме?",
         "Выбирает стратегию сопоставления и то, что считается различием.", "rules", True),
        ("Какие изменения на уровне пунктов достаточно существенны для отчёта (любая правка или только содержательные)?",
         "Позволяет не утонуть в выводе из-за тривиальных различий в формулировках.", "rules", False),
    ),
))

register(AgentArchetype(
    key="doc_generation",
    title="Подготовка / генерация документов",
    description="Генерация или редактирование документа из входных данных и шаблона.",
    runtime_strategy="generate",
    output_kind=OutputKind.NEW_DOCUMENT,
    capabilities=(_C.LLM, _C.DOC_EXTRACTION, _C.RULES, _C.LOGGING, _C.DRY_RUN),
    default_fields=(),
    clarifiers=(
        ("Предоставьте шаблон / каркас и отметьте каждое переменное место для заполнения.",
         "Генерация без фиксированного шаблона непроверяема и небезопасна в юридическом применении.", "io", True),
        ("Какие входные данные подаются в какие места, и что обязательно, а что необязательно?",
         "Определяет соответствие вход→выход, которое агент должен соблюдать.", "io", True),
        ("Есть ли пункты, которые нельзя изменять или удалять?",
         "В юридических шаблонах обычно есть неизменяемые типовые формулировки.", "rules", True),
    ),
))

register(AgentArchetype(
    key="chat_assistant",
    title="Юридический чат-ассистент",
    description="Ответы на вопросы по набору документов / базе знаний.",
    runtime_strategy="chat",
    output_kind=OutputKind.CHAT,
    capabilities=(_C.LLM, _C.RAG, _C.DOC_EXTRACTION, _C.CONFIDENCE, _C.LOGGING),
    input_arity=0,
    default_fields=(),
    clarifiers=(
        ("На какие источники знаний может опираться ассистент, и может ли он отвечать за их пределами?",
         "Область обоснования контролирует галлюцинации и ответственность.", "rules", True),
        ("Должен ли каждый ответ ссылаться на исходные фрагменты-источники?",
         "Ссылки обычно обязательны для юридической помощи.", "io", True),
        ("Какие темы или действия выходят за рамки (например, дача юридических консультаций)?",
         "Ограничители должны быть явными.", "rules", True),
    ),
))

register(AgentArchetype(
    key="workflow",
    title="Дело / многошаговый рабочий процесс",
    description="Оркестрация многошагового конвейера по делу (извлечение → проверка → отчёт).",
    runtime_strategy="workflow",
    output_kind=OutputKind.REPORT,
    capabilities=(_C.LLM, _C.DOC_EXTRACTION, _C.OCR, _C.ROUTING, _C.RULES, _C.JSON_OUTPUT,
                  _C.SCHEMA_VALIDATION, _C.CONFIDENCE, _C.MULTI_STEP, _C.LOGGING, _C.DRY_RUN),
    default_fields=(
        FieldSpec(name="case_summary", type="string", required=True, description="Краткое изложение дела."),
        FieldSpec(name="extracted", type="object", required=True, description="Структурированные данные, извлечённые из дела."),
        FieldSpec(name="checks", type="array", required=True, description="Результаты каждого правила/проверки."),
    ),
    clarifiers=(
        ("Перечислите упорядоченные шаги рабочего процесса, а также вход и выход каждого шага.",
         "Рабочий процесс необходимо декомпозировать, прежде чем его можно будет построить или протестировать.", "io", True),
        ("Какие шаги могут остановить конвейер (жёсткие сбои), а какие только аннотируют его?",
         "Определяет распространение ошибки против частичного результата по шагам.", "rules", True),
        ("Как документы маршрутизируются по типу, и что происходит с нераспознанным типом?",
         "Пробелы в маршрутизации — частый скрытый сбой.", "rules", False),
    ),
))
