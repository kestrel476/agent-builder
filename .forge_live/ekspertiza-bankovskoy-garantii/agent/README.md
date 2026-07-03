# Экспертиза банковской гарантии

Сгенерировано **Legal Agent Forge** (архетип: `checklist`, стратегия: `checklist_review`).

## Что делает
Проведение правовой экспертизы банковской гарантии путем сравнения условий текста гарантии с контрольным списком (чек-листом).

## Вход
- `document` (document)

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
См. `contract.schema.json`. Обязательные поля: results, overall_position.

## Сопровождение
Правьте `prompt.md` (поведение), `contract.schema.json` (форма вывода),
`agent.yaml` (стратегия/пороги), `rules.json` (бизнес-правила). Python-точка
входа стабильна и редко требует изменений.

> Возможности: llm, doc_extraction, ocr, rules, json_output, multi_step, logging
