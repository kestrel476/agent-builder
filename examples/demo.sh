#!/usr/bin/env bash
# Сквозная офлайн-демонстрация Legal Agent Forge.
# Собирает, тестирует, дорабатывает и упаковывает агента извлечения данных
# из договоров — без сети.
set -euo pipefail

cd "$(dirname "$0")/.."
export FORGE_HOME="$(mktemp -d)/forge-demo"
FORGE=(python3 -m forge --offline --no-color)

echo "### 1. new  (приём + критики + определение архетипа)"
"${FORGE[@]}" new --from examples/instructions_nda.txt --name "Извлечение условий договора"

echo; echo "### 2. clarify --auto  (записать явные допущения, не угадывать молча)"
"${FORGE[@]}" clarify --auto

echo; echo "### 3. contract + дополнительные поля"
"${FORGE[@]}" contract >/dev/null
"${FORGE[@]}" field \
  --add "total_amount:string:false:Общая стоимость договора / встречное предоставление" \
  --add "governing_law:string:false:Применимое право" >/dev/null
"${FORGE[@]}" build

echo; echo "### 4. добавить тест-кейс и запустить цикл тест/доработка"
"${FORGE[@]}" addtest --name northwind \
  --text "Agreement between Northwind Traders Inc and Contoso Pharmaceuticals LLC, effective February 14, 2024, total USD 75,000, governed by the laws of California." \
  --expect '{"parties":["Northwind Traders Inc","Contoso Pharmaceuticals LLC"],"governing_law":"California"}'
"${FORGE[@]}" test --refine

echo; echo "### 5. упаковка"
"${FORGE[@]}" package

echo; echo "### 6. запустить упакованного агента на образце документа"
"${FORGE[@]}" run examples/sample_contract.txt --debug

echo; echo "### артефакты в: $FORGE_HOME"
find "$FORGE_HOME" -maxdepth 3 -type f | sort
