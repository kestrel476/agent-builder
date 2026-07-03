# Детектор очевидных рисков ПРКО

Сгенерировано **Legal Agent Forge** (архетип: `catalog_risk_detection`, стратегия: `rag_detect`).

## Что делает
Автоматически выявлять очевидные правовые риски в кредитных документах (ПРКО), используя каталог правовых рисков банка.

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
См. `contract.schema.json`. Обязательные поля: detected_risks.

## Сопровождение
Правьте `prompt.md` (поведение), `contract.schema.json` (форма вывода),
`agent.yaml` (стратегия/пороги), `rules.json` (бизнес-правила). Python-точка
входа стабильна и редко требует изменений.

> Возможности: llm, rag, doc_extraction, ocr, json_output, schema_validation, confidence, multi_step, logging
