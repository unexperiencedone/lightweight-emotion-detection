"""Prosody MLP: speaker-independent training on RAVDESS / CREMA-D, then export + int8.

    python -m emotion_edge.speech.train --dataset ravdess --root data/ravdess --out artifacts/speech
Splits are by *actor* (never by utterance) -- utterance-level splits leak speaker identity and inflate accuracy by 10-20 pts.
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from emotion_edge.labels import SPEECH_LABELS
from emotion_edge.common.metrics import classification_report, fit_temperature
from emotion_edge.common import onnx_utils as ou
from emotion_edge.speech import features as F

_RAV = {"01": "neutral", "02": "calm", "03": "happy", "04": "sad", "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"}
_CREMA = {"NEU": "neutral", "HAP": "happy", "SAD": "sad", "ANG": "angry", "FEA": "fearful", "DIS": "disgust"}


def list_files(dataset, root):
    items = []
    for p in sorted(Path(root).rglob("*.wav")):
        if dataset == "ravdess":                 # 03-01-05-01-02-01-12.wav ; last field = actor id
            parts = p.stem.split("-")
            items.append((p, _RAV[parts[2]], int(parts[-1])))
        else:                                    # 1001_DFA_ANG_XX.wav ; first field = actor id
            parts = p.stem.split("_")
            if parts[2] in _CREMA:
                items.append((p, _CREMA[parts[2]], int(parts[0])))
    return items


def actor_split(actors, seed=0, val=0.15, test=0.15):
    u = np.array(sorted(set(actors)))
    rng = np.random.default_rng(seed)
    rng.shuffle(u)
    nt, nv = max(1, int(len(u) * test)), max(1, int(len(u) * val))
    return set(u[:nt]), set(u[nt:nt + nv])


class ProsodyMLP(nn.Module):
    def __init__(self, d_in, n_out=len(SPEECH_LABELS), h=128, p=0.3):
        super().__init__()
        self.register_buffer("mu", torch.zeros(d_in))
        self.register_buffer("sd", torch.ones(d_in))
        self.net = nn.Sequential(nn.Linear(d_in, h), nn.ReLU(), nn.Dropout(p), nn.Linear(h, h), nn.ReLU(), nn.Dropout(p), nn.Linear(h, n_out))

    def forward(self, x):
        return self.net((x - self.mu) / self.sd)


def fit(Xtr, ytr, Xva, yva, epochs=200, lr=2e-3, wd=1e-3, seed=0, n_out=len(SPEECH_LABELS)):
    torch.manual_seed(seed)
    m = ProsodyMLP(Xtr.shape[1], n_out)
    m.mu.copy_(torch.from_numpy(Xtr.mean(0)))
    m.sd.copy_(torch.from_numpy(Xtr.std(0) + 1e-6))
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    Xt, yt, Xv = map(torch.from_numpy, (Xtr, ytr, Xva))
    best, state = -1, None
    for ep in range(epochs):
        m.train()
        for idx in torch.randperm(len(Xt)).split(64):
            x = Xt[idx] + 0.05 * torch.randn_like(Xt[idx]) * m.sd   # feature-noise augmentation
            loss = nn.functional.cross_entropy(m(x), yt[idx], label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            acc = (m(Xv).argmax(1).numpy() == yva).mean()
        if acc > best:
            best, state = acc, {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(state)
    return m.eval(), float(best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["ravdess", "cremad"], required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="artifacts/speech")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    items = list_files(a.dataset, a.root)
    labels = SPEECH_LABELS
    X, y, actors = [], [], []
    for p, lab, act in items:
        feats, names = F.prosody_features(F.load_audio(p))
        X.append(feats); y.append(labels.index(lab)); actors.append(act)
    X, y, actors = np.stack(X), np.array(y), np.array(actors)
    test_a, val_a = actor_split(actors)
    te, va = np.isin(actors, list(test_a)), np.isin(actors, list(val_a))
    tr = ~(te | va)
    m, vacc = fit(X[tr], y[tr], X[va], y[va])
    with torch.no_grad():
        vl, tl = m(torch.from_numpy(X[va])).numpy(), m(torch.from_numpy(X[te])).numpy()
    T, cal = fit_temperature(vl, y[va])
    res = {"dataset": a.dataset, "n": len(y), "split_by": "actor", "val_acc": vacc,
           "fp32": classification_report(y[te], tl.argmax(1), labels), "calibration": {"temperature": T, **cal},
           "params": sum(p.numel() for p in m.parameters())}
    np.savez(out / "logits.npz", val=vl, test=tl, y_val=y[va], y_test=y[te])
    ex = torch.randn(1, X.shape[1])
    ou.export_onnx(m, (ex,), out / "prosody_fp32.onnx", ["feat"], ["logits"], {"feat": {0: "b"}, "logits": {0: "b"}})
    ou.quantize_dynamic_int8(out / "prosody_fp32.onnx", out / "prosody_int8.onnx")
    sess = ou.ort_session(out / "prosody_int8.onnx")
    q = sess.run(None, {"feat": X[te]})[0]
    res["int8"] = classification_report(y[te], q.argmax(1), labels)
    res["int8"]["size_mb"] = ou.file_mb(out / "prosody_int8.onnx")
    (out / "results.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: res[k] for k in ("val_acc", "params")}), "test fp32", res["fp32"]["accuracy_rounded_pct"], "int8", res["int8"]["accuracy_rounded_pct"])


if __name__ == "__main__":
    main()
