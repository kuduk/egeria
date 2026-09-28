#!/usr/bin/env bash
# Baseline F0 zero-shot su typed-decisions per uno o più modelli.
#
#   scripts/baseline.sh Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-2B
#   EXTRA="--quantize 4bit" scripts/baseline.sh Qwen/Qwen3.5-4B
#
# Per ogni modello:
# 1. predizioni sul train (CAL_CASES casi equispaziati), solo per la calibrazione;
# 2. predizioni sull'intero test;
# 3. temperature per tipo e per layer di uscita, fittate sul train;
# 4. report sul test, con T=1 e con le temperature fittate.
set -euo pipefail
cd "$(dirname "$0")/.."

CAL_CASES="${CAL_CASES:-300}"
PERMUTATIONS="${PERMUTATIONS:-2}"
EXTRA="${EXTRA:-}"
EGERIA=.venv/bin/egeria

for model in "$@"; do
  name="$(basename "$model")${TAG:-}"
  dir="runs/$name"
  mkdir -p "$dir"
  echo "=== $model -> $dir"
  # Il file .meta.json viene scritto solo a predizione completata: se esiste, il passo si salta.
  [ -f "$dir/train.meta.json" ] || $EGERIA predict --model "$model" $EXTRA --permutations "$PERMUTATIONS" \
    --exits blocks --split train --limit "$CAL_CASES" --out "$dir/train.jsonl"
  [ -f "$dir/test.meta.json" ] || $EGERIA predict --model "$model" $EXTRA --permutations "$PERMUTATIONS" \
    --exits blocks --split test --out "$dir/test.jsonl"
  $EGERIA calibrate --predictions "$dir/train.jsonl" --out "$dir/temperature.json"
  $EGERIA evaluate --predictions "$dir/test.jsonl" --prior "$dir/train.jsonl" --depth --report "$dir/report-T1.json" | tee "$dir/report-T1.txt"
  $EGERIA evaluate --predictions "$dir/test.jsonl" --temperatures "$dir/temperature.json" --prior "$dir/train.jsonl" --depth \
    --report "$dir/report-cal.json" | tee "$dir/report-cal.txt"
done
