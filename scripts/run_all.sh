#!/usr/bin/env bash
# Scarica i modelli e lancia la baseline F0 in sequenza (ripartibile).
# Uso, staccato dalla sessione:  setsid nohup scripts/run_all.sh > runs/run_all.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
MODELS="${MODELS:-Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-0.8B-Base Qwen/Qwen3.5-2B Qwen/Qwen3.5-2B-Base}"
for model in $MODELS; do
  .venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('$model', allow_patterns=['*.json','*.safetensors','*.txt','*.jinja','tokenizer*','merges.txt','vocab.json'])
" 2>&1 | grep -v -E "Fetching|Warning"
  scripts/baseline.sh "$model"
done
echo "RUN_ALL_DONE"
