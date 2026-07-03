# Проверка рисков аренды

Сгенерировано **Legal Agent Forge** (архетип: `risk_check`, стратегия: `extract`).

## Что делает
Проверять коммерческие договоры аренды и помечать рисковые условия: автопродление, ответственность арендатора за капитальный ремонт, личное поручительство, неограниченное повышение арендной платы. Сформировать findings с общим уровнем риска и цитатой условия. Вывод JSON.

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
См. `contract.schema.json`. Обязательные поля: findings, overall_risk.

## Сопровождение
Правьте `prompt.md` (поведение), `contract.schema.json` (форма вывода),
`agent.yaml` (стратегия/пороги), `rules.json` (бизнес-правила). Python-точка
входа стабильна и редко требует изменений.

> Возможности: llm, doc_extraction, ocr, rules, json_output, confidence, logging, multi_step
