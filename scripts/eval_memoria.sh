#!/usr/bin/env bash
# Effetto della memoria (ricordi richiamati nel prompt) su typed-decisions.
#   MODEL=Qwen/Qwen3.5-2B-Base RECALL=3 scripts/eval_memoria.sh
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="${MODEL:-Qwen/Qwen3.5-2B-Base}"
RECALL="${RECALL:-3}"
PERMUTATIONS="${PERMUTATIONS:-2}"
name="$(basename "$MODEL")"
store="runs/memoria-$name"
dir="runs/$name-memoria-k$RECALL"
S=.venv/bin/egeria
mkdir -p "$dir"
[ -f "$store/memories.jsonl" ] || $S memory build --model "$MODEL" --memory "$store" --split train
# train: il caso stesso è escluso dal richiamo (leave-one-out)
[ -f "$dir/train.meta.json" ] || $S predict --model "$MODEL" --permutations "$PERMUTATIONS" --memory "$store" \
  --recall "$RECALL" --inject --split train --limit 300 --out "$dir/train.jsonl"
[ -f "$dir/test.meta.json" ] || $S predict --model "$MODEL" --permutations "$PERMUTATIONS" --memory "$store" \
  --recall "$RECALL" --inject --split test --out "$dir/test.jsonl"
$S calibrate --predictions "$dir/train.jsonl" --out "$dir/temperature.json"
$S evaluate --predictions "$dir/test.jsonl" --temperatures "$dir/temperature.json" --prior "$dir/train.jsonl" \
  --report "$dir/report-cal.json" | tee "$dir/report-cal.txt"
$S paired --a "runs/$name/test.jsonl" --temperatures-a "runs/$name/temperature.json" \
  --b "$dir/test.jsonl" --temperatures-b "$dir/temperature.json" | tee "$dir/paired.txt"
echo "EVAL_MEMORIA_DONE"
