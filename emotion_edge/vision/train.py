"""Train + export the face model on FER2013-format data (fer2013.csv: emotion,pixels,Usage) or an ImageFolder-style dir.

    python -m emotion_edge.vision.train --csv data/fer2013.csv --out artifacts/face
Evaluation uses FER2013's PublicTest for model selection and PrivateTest for the reported number.
Inference-time face localisation: OpenCV Haar cascade (ships with opencv, ~0.5 ms) -> crop -> 48x48 gray.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from emotion_edge.labels import FACE_LABELS
from emotion_edge.common.metrics import classification_report, fit_temperature
from emotion_edge.common import onnx_utils as ou
from emotion_edge.vision.model import MiniXception


def load_fer_csv(path):
    import pandas as pd
    df = pd.read_csv(path)
    X = np.stack([np.fromstring(s, sep=" ", dtype=np.uint8).reshape(48, 48) for s in df.pixels])
    # FER2013 ids: 0 angry 1 disgust 2 fear 3 happy 4 sad 5 surprise 6 neutral == FACE_LABELS order
    return X, df.emotion.values.astype(np.int64), df.Usage.values


def detect_and_crop(bgr, size=48):
    """Largest Haar-cascade face -> (crop float32 [0,1], detector quality in [0,1]); None if no face."""
    import cv2
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    cas = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cas.detectMultiScale(g, 1.1, 5, minSize=(40, 40))
    if len(faces) == 0:
        return None, 0.0
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    crop = cv2.resize(g[y:y + h, x:x + w], (size, size)).astype(np.float32) / 255
    quality = float(np.clip(min(w, h) / 120.0, 0.2, 1.0))      # tiny faces -> less reliable
    return crop, quality


def augment(x):
    """x: (B,1,48,48) tensor. random flip, small shift, brightness."""
    flip = torch.rand(x.size(0)) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    dx, dy = np.random.randint(-3, 4, 2)
    x = torch.roll(x, (int(dy), int(dx)), (2, 3))
    return (x * (0.8 + 0.4 * torch.rand(x.size(0), 1, 1, 1))).clamp(0, 1)


def fit(Xtr, ytr, Xva, yva, epochs=60, lr=3e-3, seed=0, aug=True):
    torch.manual_seed(seed)
    m = MiniXception()
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * ((len(Xtr) + 127) // 128))
    Xt, yt = torch.from_numpy(Xtr).float()[:, None] / 255, torch.from_numpy(ytr)
    Xv = torch.from_numpy(Xva).float()[:, None] / 255
    best, state = -1, None
    for ep in range(epochs):
        m.train()
        for idx in torch.randperm(len(Xt)).split(128):
            loss = F.cross_entropy(m(augment(Xt[idx]) if aug else Xt[idx]), yt[idx], label_smoothing=0.1)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        m.eval()
        with torch.no_grad():
            acc = float((m(Xv).argmax(1).numpy() == yva).mean())
        if acc > best:
            best, state = acc, {k: v.clone() for k, v in m.state_dict().items()}
        print(f"epoch {ep + 1}/{epochs} val_acc {acc:.4f}", flush=True)
    m.load_state_dict(state)
    return m.eval(), best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="artifacts/face")
    ap.add_argument("--epochs", type=int, default=60)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    X, y, use = load_fer_csv(a.csv)
    tr, va, te = use == "Training", use == "PublicTest", use == "PrivateTest"
    m, vacc = fit(X[tr], y[tr], X[va], y[va], a.epochs)
    f = lambda a_: torch.from_numpy(a_).float()[:, None] / 255
    with torch.no_grad():
        vl, tl = m(f(X[va])).numpy(), m(f(X[te])).numpy()
    T, cal = fit_temperature(vl, y[va])
    res = {"val_acc": vacc, "params": sum(p.numel() for p in m.parameters()), "calibration": {"temperature": T, **cal},
           "fp32": classification_report(y[te], tl.argmax(1), FACE_LABELS)}
    np.savez(out / "logits.npz", val=vl, test=tl, y_val=y[va], y_test=y[te])
    ex = torch.rand(1, 1, 48, 48)
    ou.export_onnx(m, (ex,), out / "face_fp32.onnx", ["img"], ["logits"], {"img": {0: "b"}, "logits": {0: "b"}})
    calib = [f(X[tr][i:i + 1]).numpy() for i in range(0, 200)]
    ou.quantize_static_int8(out / "face_fp32.onnx", out / "face_int8.onnx", calib, "img")
    q = np.concatenate([ou.ort_session(out / "face_int8.onnx").run(None, {"img": f(X[te][i:i + 64]).numpy()})[0] for i in range(0, te.sum(), 64)])
    res["int8"] = classification_report(y[te], q.argmax(1), FACE_LABELS)
    res["int8"]["size_mb"] = ou.file_mb(out / "face_int8.onnx")
    (out / "results.json").write_text(json.dumps(res, indent=2, default=float))
    print("test fp32", res["fp32"]["accuracy_rounded_pct"], "int8", res["int8"]["accuracy_rounded_pct"])


if __name__ == "__main__":
    main()
