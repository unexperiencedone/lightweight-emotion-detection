#!/usr/bin/env bash
# Full real run of the text track. Needs huggingface.co reachable (dataset + distilbert-base-uncased) -- or
#   DATA_DIR=<dir with train/validation/test.parquet|jsonl|csv>  MODEL=<local distilbert dir>  for offline use.
# CPU: ~5 min/epoch on 4 cores. GPU (Colab T4): ~30 s/epoch.
set -euo pipefail
export PYTHONPATH="$PWD"
MODEL=${MODEL:-distilbert-base-uncased}
ARGS=(--model "$MODEL")
[ -n "${DATA_DIR:-}" ] && ARGS+=(--data-dir "$DATA_DIR")

# 1) small validation-selected sweep (selection uses the *validation* split only; test is read once per run for reporting)
best=""; best_acc=0
for lr in 3e-5 5e-5; do for ep in 3 4; do
  out=artifacts/sweep/lr${lr}_ep${ep}
  python -m emotion_edge.text.pipeline "${ARGS[@]}" --lr "$lr" --epochs "$ep" --gate 101 --out "$out"   # gate=101: sweep never quantizes
  acc=$(python -c "import json;h=json.load(open('$out/results.json'))['train']['history'];print(max(e['val_acc'] for e in h))")
  if python -c "import sys;sys.exit(0 if $acc>$best_acc else 1)"; then best=$out; best_acc=$acc; best_lr=$lr; best_ep=$ep; fi
done; done
echo "best val acc $best_acc at lr=$best_lr epochs=$best_ep"

# 2) final run with the chosen setting: gate at 90 % rounded test accuracy, then prune/quantize/export
python -m emotion_edge.text.pipeline "${ARGS[@]}" --lr "$best_lr" --epochs "$best_ep" --gate 90 --out artifacts/text

# 3) edge simulation of the deployed variants + report
python scripts/run_edge_bench.py --kind text --out results/edge_text.json \
  fp32=artifacts/text/model_fp32.onnx int8=artifacts/text/model_int8.onnx \
  pruned_int8_emb8=artifacts/text/model_pruned_int8_emb8.onnx
python -m emotion_edge.report
