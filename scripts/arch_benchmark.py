"""Latency/size benchmark of the *architecture* with RANDOM weights (valid for speed/size, says nothing about accuracy).
Used while pretrained weights / dataset are unavailable, and to cross-check the real model's numbers."""
import sys, torch
from transformers import DistilBertConfig, DistilBertForSequenceClassification
from emotion_edge.text.model import ExportWrapper
from emotion_edge.common import onnx_utils as ou
from emotion_edge.edge.bench import run_all

out = sys.argv[1] if len(sys.argv) > 1 else "artifacts/arch"
torch.manual_seed(0)
m = DistilBertForSequenceClassification(DistilBertConfig(num_labels=6)).eval()  # 6 layers, 768d, 30522 vocab
print("params", ou.count_params(m))
ex = (torch.randint(5, 1000, (1, 32)), torch.ones(1, 32, dtype=torch.long))
axes = {"input_ids": {0: "b", 1: "s"}, "attention_mask": {0: "b", 1: "s"}, "logits": {0: "b"}}
ou.export_onnx(ExportWrapper(m), ex, f"{out}/fp32.onnx", ["input_ids", "attention_mask"], ["logits"], axes)
ou.quantize_dynamic_int8(f"{out}/fp32.onnx", f"{out}/int8.onnx")
# vocab-pruned variant at a hypothetical 15k kept tokens (size/latency only)
import torch.nn as nn
m.distilbert.embeddings.word_embeddings = nn.Embedding(15000, 768)
ou.export_onnx(ExportWrapper(m), ex, f"{out}/pruned15k_fp32.onnx", ["input_ids", "attention_mask"], ["logits"], axes)
ou.quantize_dynamic_int8(f"{out}/pruned15k_fp32.onnx", f"{out}/pruned15k_int8.onnx")
from emotion_edge.text.model import quantize_embeddings
quantize_embeddings(m)
ou.export_onnx(ExportWrapper(m), ex, f"{out}/tmp.onnx", ["input_ids", "attention_mask"], ["logits"], axes)
ou.quantize_dynamic_int8(f"{out}/tmp.onnx", f"{out}/pruned15k_int8_emb8.onnx")
import os; os.remove(f"{out}/tmp.onnx")
for n in ("fp32", "pruned15k_int8_emb8", "int8", "pruned15k_fp32", "pruned15k_int8"):
    print(n, round(ou.file_mb(f"{out}/{n}.onnx"), 1), "MB")
