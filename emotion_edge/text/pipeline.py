"""End-to-end text pipeline: train -> evaluate -> gate -> prune vocab -> export -> int8 -> evaluate -> save.

    python -m emotion_edge.text.pipeline --out artifacts/text            # real run (needs HF access or --data-dir + --model-dir)
    python -m emotion_edge.text.pipeline --smoke --out artifacts/smoke   # code-path check on SYNTHETIC data, tiny random model
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch

from emotion_edge.labels import TEXT_LABELS
from emotion_edge.common.metrics import classification_report, fit_temperature
from emotion_edge.common import onnx_utils as ou
from emotion_edge.text import data as D, model as M


def onnx_logits(path, tok, texts, bs=64):
    sess = ou.ort_session(path, threads=4)
    order = np.argsort([len(t) for t in texts])
    out = np.zeros((len(texts), 6), dtype=np.float32)
    for i in range(0, len(texts), bs):
        sel = order[i:i + bs]
        ids, mask = tok([texts[j] for j in sel])
        out[sel] = sess.run(None, {"input_ids": ids, "attention_mask": mask})[0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/text")
    ap.add_argument("--model", default="distilbert-base-uncased", help="HF id or local dir")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--teacher", default=None, help="optional fine-tuned teacher dir for knowledge distillation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gate", type=float, default=90.0, help="min rounded test accuracy (%%) required to quantize")
    ap.add_argument("--max-quant-drop", type=float, default=1.0, help="max accuracy drop (pts) accepted for a quantized variant")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    res = {"synthetic": a.smoke, "args": vars(a), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    log = lambda *x: print(*x, flush=True)

    from transformers import AutoTokenizer, DistilBertConfig, DistilBertForSequenceClassification, BertTokenizerFast
    if a.smoke:
        frames = D.synthetic_dair()
        vocab = D.write_synthetic_vocab(str(out / "vocab.txt"), frames)
        hf_tok = BertTokenizerFast(vocab={w: i for i, w in enumerate(Path(vocab).read_text().split(chr(10)))}, do_lower_case=True)
        cfg = DistilBertConfig(vocab_size=hf_tok.vocab_size, dim=64, hidden_dim=128, n_layers=2, n_heads=2,
                               max_position_embeddings=128, num_labels=6)
        model = DistilBertForSequenceClassification(cfg)
    else:
        frames = D.load_dair(a.data_dir)
        hf_tok = AutoTokenizer.from_pretrained(a.model)
        model = DistilBertForSequenceClassification.from_pretrained(a.model, num_labels=6)
    tr, va, te = frames["train"], frames["validation"], frames["test"]
    res["data"] = {"train": len(tr), "val": len(va), "test": len(te),
                   "class_counts_train": np.bincount(tr.label.values, minlength=6).tolist()}
    tok = M.Tok(hf_tok, a.max_len)
    res["data"]["truncation_rate_train"] = tok.truncation_rate(list(tr.text))
    res["params_fp32"] = ou.count_params(model)
    log("params:", res["params_fp32"])

    teacher = None
    if a.teacher:
        teacher = DistilBertForSequenceClassification.from_pretrained(a.teacher).eval()
    t0 = time.time()
    model, hist = M.train(model, tok, tr, va, epochs=a.epochs, lr=a.lr, bs=a.bs, seed=a.seed,
                          label_smoothing=a.label_smoothing, teacher=teacher, log=log)
    res["train"] = {"history": hist, "wall_sec": time.time() - t0}
    model.save_pretrained(out / "torch_fp32")
    hf_tok.save_pretrained(out / "torch_fp32")

    yv, yt = va.label.values, te.label.values
    vl = M.predict_logits(model, tok, list(va.text))
    tl = M.predict_logits(model, tok, list(te.text))
    T, calib = fit_temperature(vl, yv)
    res["calibration"] = {"temperature": T, **calib}
    np.savez(out / "logits.npz", val=vl, test=tl, y_val=yv, y_test=yt)
    variants = {"torch_fp32": classification_report(yt, tl.argmax(1), TEXT_LABELS)}
    base_acc = variants["torch_fp32"]["accuracy_rounded_pct"]
    res["gate"] = {"threshold_pct": a.gate, "test_acc_rounded_pct": base_acc, "passed": base_acc >= a.gate}
    log(f"torch fp32 test acc {base_acc}%  macroF1 {variants['torch_fp32']['macro_f1']:.4f}  gate={'PASS' if base_acc >= a.gate else 'FAIL'}")

    if res["gate"]["passed"] or a.smoke:
        if not res["gate"]["passed"]:
            log("(smoke) gate failed but continuing so the quantization code path is exercised")
        model.eval()
        ex = (torch.randint(5, min(50, hf_tok.vocab_size), (1, 16)), torch.ones(1, 16, dtype=torch.long))
        axes = {"input_ids": {0: "b", 1: "s"}, "attention_mask": {0: "b", 1: "s"}, "logits": {0: "b"}}
        specs = {}
        # A/B: full vocab
        p32 = out / "model_fp32.onnx"
        ou.export_onnx(M.ExportWrapper(model), ex, p32, ["input_ids", "attention_mask"], ["logits"], axes)
        p8 = out / "model_int8.onnx"
        ou.quantize_dynamic_int8(p32, p8)
        specs["onnx_fp32"], specs["onnx_int8"] = (p32, tok), (p8, tok)
        # C/D: vocabulary-pruned (kept = tokens in train+val; test OOV -> [UNK], so no test leakage)
        remap, kept = M.build_remap(hf_tok, list(tr.text) + list(va.text), hf_tok.vocab_size)
        import copy
        pm = M.prune_embeddings(copy.deepcopy(model), kept).eval()
        ptok = M.Tok(hf_tok, a.max_len, remap)
        pp32, pp8 = out / "model_pruned_fp32.onnx", out / "model_pruned_int8.onnx"
        ou.export_onnx(M.ExportWrapper(pm), ex, pp32, ["input_ids", "attention_mask"], ["logits"], axes)
        ou.quantize_dynamic_int8(pp32, pp8)
        np.save(out / "vocab_remap.npy", remap)
        # E: pruned + int8 encoder + int8 embedding table (smallest)
        qm = M.quantize_embeddings(copy.deepcopy(pm)).eval()
        pe8 = out / "model_pruned_int8_emb8.onnx"
        ou.export_onnx(M.ExportWrapper(qm), ex, out / "tmp_emb8.onnx", ["input_ids", "attention_mask"], ["logits"], axes)
        ou.quantize_dynamic_int8(out / "tmp_emb8.onnx", pe8)
        (out / "tmp_emb8.onnx").unlink()
        specs["onnx_pruned_fp32"], specs["onnx_pruned_int8"] = (pp32, ptok), (pp8, ptok)
        specs["onnx_pruned_int8_emb8"] = (pe8, ptok)
        res["vocab_pruning"] = {"kept": len(kept), "original": int(hf_tok.vocab_size),
                                "params_after": ou.count_params(pm)}
        for name, (path, tk) in specs.items():
            lg = onnx_logits(path, tk, list(te.text))
            r = classification_report(yt, lg.argmax(1), TEXT_LABELS)
            r.update(size_mb=ou.file_mb(path), drop_vs_fp32_pts=base_acc - r["accuracy_rounded_pct"],
                     agreement_with_torch=float((lg.argmax(1) == tl.argmax(1)).mean()), path=str(path))
            variants[name] = r
            log(f"{name:18s} acc {r['accuracy_rounded_pct']}%  size {r['size_mb']:.1f}MB  drop {r['drop_vs_fp32_pts']:.1f}pt")
        ok = {k: v for k, v in variants.items() if k.startswith("onnx") and v["drop_vs_fp32_pts"] <= a.max_quant_drop}
        # deployment pick: smallest variant that stays within the accuracy budget
        res["deploy_choice"] = min(ok, key=lambda k: ok[k]["size_mb"]) if ok else "onnx_fp32"
        log("deploy choice:", res["deploy_choice"])
    else:
        res["skipped"] = f"quantization skipped: rounded test accuracy {base_acc}% < gate {a.gate}%"
        log(res["skipped"])
    res["variants"] = variants
    (out / "results.json").write_text(json.dumps(res, indent=2, default=float))
    log("wrote", out / "results.json")


if __name__ == "__main__":
    main()
