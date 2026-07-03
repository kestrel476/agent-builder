# Ревью пакета сделки

Сгенерировано **Legal Agent Forge** (архетип: `document_package`, стратегия: `package_review`).

## Что делает
Проверка комплекта документов и рисков по кредитной сделке

## Вход
- `document_a` (document)
- `document_b` (document)

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
См. `contract.schema.json`. Обязательные поля: verdict, documents.

## Сопровождение
Правьте `prompt.md` (поведение), `contract.schema.json` (форма вывода),
`agent.yaml` (стратегия/пороги), `rules.json` (бизнес-правила). Python-точка
входа стабильна и редко требует изменений.

> Возможности: llm, doc_extraction, ocr, routing, multi_step, json_output, schema_validation, logging
