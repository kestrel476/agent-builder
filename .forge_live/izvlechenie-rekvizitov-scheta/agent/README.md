# Извлечение реквизитов счёта

Сгенерировано **Legal Agent Forge** (архетип: `json_extraction`, стратегия: `extract`).

## Что делает
Извлечение ключевых реквизитов из счетов на оплату и представление их в формате JSON.

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
См. `contract.schema.json`. Обязательные поля: supplier_name, supplier_inn, invoice_number, invoice_date, total_amount.

## Сопровождение
Правьте `prompt.md` (поведение), `contract.schema.json` (форма вывода),
`agent.yaml` (стратегия/пороги), `rules.json` (бизнес-правила). Python-точка
входа стабильна и редко требует изменений.

> Возможности: llm, doc_extraction, ocr, json_output, schema_validation, confidence, logging
