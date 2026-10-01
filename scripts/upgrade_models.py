"""Upgrade the voice and face models with PRETRAINED encoders, evaluated exactly like the first CREMA-D study
(same 61/15/15 actor split, choices on val actors, test reported once).

  voice:  frozen self-supervised speech encoder (DistilHuBERT for the edge, WavLM-base-plus as reference)
          + SUPERB-style head: softmax-weighted sum of layers -> mean/std pooled -> MLP
  face:   ImageNet-pretrained MobileNetV3-large fine-tuned on 112x112 grayscale face crops

    python scripts/upgrade_models.py voice --root <cremad> --feats <feats> --ssl ssl_distilhubert.npz --tag distilhubert
    python scripts/upgrade_models.py face-extract --root <cremad> --feats <feats>
    python scripts/upgrade_models.py face --root <cremad> --feats <feats>
    python scripts/upgrade_models.py fuse --root <cremad> --feats <feats> --voice distilhubert --face mnv3
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scipy.special import softmax, log_softmax
from sklearn.metrics import f1_score

from emotion_edge.labels import SPEECH_LABELS, FACE_LABELS
from emotion_edge.realdata import cremad as C
from emotion_edge.common.metrics import fit_temperature

OUT = Path("results/upgrade.json")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def save(key, val):
    r = json.loads(OUT.read_text()) if OUT.exists() else {}
    r[key] = val
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(r, indent=2, default=float))


def report(df, L6, te, human_mod, T):
    """L6: logits over the 6 CREMA-D classes (A D F H N S)."""
    p = softmax(L6 / T, 1); pred = p.argmax(1); y = df.y.to_numpy(); hum = df[f"h_{human_mod}_top"].to_numpy()
    return {"test_acc": float((pred[te] == y[te]).mean()), "test_macro_f1": float(f1_score(y[te], pred[te], average="macro")),
            "human_acc_same_clips": float((hum[te] == y[te]).mean()), "temperature": T,
            "per_class_recall": {C.NAMES[k]: float((pred[te][y[te] == k] == k).mean()) for k in range(6)},
            "confusion": np.bincount(y[te] * 6 + pred[te], minlength=36).reshape(6, 6).tolist()}


# ------------------------------------------------------------------------------------------------ voice
class SSLHead(nn.Module):
    """Softmax-weighted layer sum + MLP over [mean ; std] pooled features. Output: 8 SPEECH_LABELS logits."""

    def __init__(self, n_layers, dim, n_out=len(SPEECH_LABELS), h=256, p=0.3):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(n_layers))
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Dropout(p), nn.Linear(dim, h), nn.ReLU(), nn.Dropout(p), nn.Linear(h, n_out))

    def forward(self, x):                         # x: (B, layers, dim)
        z = (softmax_t(self.w)[None, :, None] * x).sum(1)
        return self.mlp(self.norm(z))


def softmax_t(w):
    return torch.softmax(w, 0)


def train_voice(a):
    df = C.metadata(a.root); df["split"] = C.actor_split(df)
    Z = np.load(Path(a.feats) / a.ssl, allow_pickle=True)   # our own extraction output
    assert list(Z["files"]) == list(df.file)
    X = torch.from_numpy(Z["X"].astype(np.float32))
    y = torch.from_numpy(C.speech_index(df.code))
    tr, va, te = [(df.split == s).to_numpy() for s in ("train", "val", "test")]
    best = None
    for seed in range(3):
        for lr in (1e-3, 3e-4):
            torch.manual_seed(seed)
            m = SSLHead(X.shape[1], X.shape[2])
            opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-2)
            bacc, bstate = -1, None
            for ep in range(60):
                m.train()
                for idx in torch.randperm(int(tr.sum())).split(64):
                    xb = X[tr][idx]
                    xb = xb + 0.02 * torch.randn_like(xb) * xb.std()           # feature noise
                    loss = F.cross_entropy(m(xb), y[tr][idx], label_smoothing=0.1)
                    opt.zero_grad(); loss.backward(); opt.step()
                m.eval()
                with torch.no_grad():
                    acc = float((m(X[va]).argmax(1) == y[va]).float().mean())
                if acc > bacc:
                    bacc, bstate = acc, {k: v.clone() for k, v in m.state_dict().items()}
            log(f"voice {a.tag} seed {seed} lr {lr}: best val {bacc:.4f}")
            if best is None or bacc > best[0]:
                best = (bacc, bstate, seed, lr)
    m = SSLHead(X.shape[1], X.shape[2]); m.load_state_dict(best[1]); m.eval()
    with torch.no_grad():
        L = m(X).numpy()
    idx6 = [SPEECH_LABELS.index(C.TO_SPEECH[c]) for c in C.CODES]
    T, cal = fit_temperature(L[va][:, idx6], df.y.to_numpy()[va])
    res = report(df, L[:, idx6], te, "voice", T)
    res.update(val_acc=best[0], seed=best[2], lr=best[3], encoder=str(Z["model"]),
               layer_weights=softmax_t(m.w).detach().numpy().round(3).tolist(), calibration=cal,
               head_params=sum(p.numel() for p in m.parameters()))
    np.save(Path(a.feats) / f"voice_{a.tag}_logits.npy", L)
    torch.save(m.state_dict(), Path(a.feats) / f"voice_{a.tag}_head.pt")
    save(f"voice_{a.tag}", res)
    log(f"voice {a.tag}: TEST acc {res['test_acc']:.4f} (humans {res['human_acc_same_clips']:.4f})")


# ------------------------------------------------------------------------------------------------ face
def face_extract(a):
    from multiprocessing import Pool
    df = C.metadata(a.root)
    global ROOT
    with Pool(4, initializer=_init, initargs=(a.root,)) as pool:
        res = pool.map(_crop112, list(df.file), chunksize=8)
    crops = np.concatenate([r[0] for r in res]); clip = np.concatenate([np.full(len(r[0]), i) for i, r in enumerate(res)])
    np.savez(Path(a.feats) / "faces112.npz", crops=crops, clip=clip, time=np.concatenate([r[1] for r in res]),
             quality=np.concatenate([r[2] for r in res]), files=np.array(df.file))
    log("faces112", crops.shape)


def _init(root):
    global ROOT
    ROOT = root


def _crop112(f):
    c, t, q, n = C.face_crops(f"{ROOT}/VideoFlash/{f}.flv", size=112)
    return c, t, q


def build_face_model():
    import timm
    return timm.create_model("mobilenetv3_large_100.ra_in1k", pretrained=True, num_classes=len(FACE_LABELS), in_chans=1,
                             drop_rate=0.2)


def face_aug(x):
    """x: (B,1,112,112) float [0,1]: random flip, random-resized-crop-ish shift/scale, brightness/contrast."""
    B = x.size(0)
    flip = torch.rand(B) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    s = 1 + 0.15 * (torch.rand(B) - 0.5)
    tx, ty = 0.1 * (torch.rand(B) - 0.5), 0.1 * (torch.rand(B) - 0.5)
    theta = torch.zeros(B, 2, 3); theta[:, 0, 0] = s; theta[:, 1, 1] = s; theta[:, 0, 2] = tx; theta[:, 1, 2] = ty
    x = F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False), padding_mode="border", align_corners=False)
    c = 0.8 + 0.4 * torch.rand(B, 1, 1, 1); b = 0.1 * (torch.rand(B, 1, 1, 1) - 0.5)
    return ((x - 0.5) * c + 0.5 + b).clamp(0, 1)


def norm(x):
    return (x - 0.5) / 0.25


def clip_logits(model, crops, clip, n, bs=256):
    Lc = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(crops), bs):
            Lc.append(model(norm(torch.from_numpy(crops[i:i + bs]).float()[:, None] / 255)).numpy())
    Lc = np.concatenate(Lc)
    L = np.zeros((n, Lc.shape[1])); cnt = np.bincount(clip, minlength=n)
    np.add.at(L, clip, log_softmax(Lc, 1)); L[cnt > 0] /= cnt[cnt > 0, None]
    return L, Lc, cnt > 0


def train_face(a):
    torch.set_num_threads(4)
    df = C.metadata(a.root); df["split"] = C.actor_split(df)
    Fz = np.load(Path(a.feats) / "faces112.npz")
    crops, clip = Fz["crops"], Fz["clip"]
    sp = df.split.to_numpy()[clip]
    yc = C.face_index(df.code)[clip]
    tr_idx = np.where(sp == "train")[0]
    va_clips = (df.split == "val").to_numpy()
    va_crop = np.where(sp == "val")[0]
    va_sub = va_crop[::3]                                              # every 3rd val frame for fast model selection
    m = build_face_model()
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.05)
    per_clip = {}
    for k in tr_idx:
        per_clip.setdefault(int(clip[k]), []).append(k)
    clips_tr = list(per_clip)
    steps = a.epochs * ((len(clips_tr) * a.frames_per_clip + 63) // 64)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.15)
    rng = np.random.default_rng(0)
    best, state = -1, None
    for ep in range(a.epochs):
        t0 = time.time()
        sel = np.concatenate([rng.choice(per_clip[c], min(a.frames_per_clip, len(per_clip[c])), replace=False) for c in clips_tr])
        rng.shuffle(sel)
        m.train()
        for i in range(0, len(sel), 64):
            b = sel[i:i + 64]
            x = face_aug(torch.from_numpy(crops[b]).float()[:, None] / 255)
            loss = F.cross_entropy(m(norm(x)), torch.from_numpy(yc[b]), label_smoothing=0.1)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        Lsub = []
        m.eval()
        with torch.no_grad():
            for i in range(0, len(va_sub), 256):
                Lsub.append(m(norm(torch.from_numpy(crops[va_sub[i:i + 256]]).float()[:, None] / 255)).numpy())
        Lsub = np.concatenate(Lsub)
        n = len(df); L = np.zeros((n, 7)); cnt = np.bincount(clip[va_sub], minlength=n)
        np.add.at(L, clip[va_sub], log_softmax(Lsub, 1))
        has = cnt > 0
        acc = float((L[has].argmax(1) == C.face_index(df.code)[has]).mean())
        log(f"face epoch {ep + 1}/{a.epochs} val clip acc {acc:.4f} ({time.time() - t0:.0f}s)")
        if acc > best:
            best, state = acc, {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(state)
    torch.save(state, Path(a.feats) / "face_mnv3.pt")
    L, Lc, has = clip_logits(m, crops, clip, len(df))
    te = (df.split == "test").to_numpy() & has
    idx6 = [FACE_LABELS.index(C.TO_FACE[c]) for c in C.CODES]
    vclip = va_clips & has
    T, cal = fit_temperature(L[vclip][:, idx6], df.y.to_numpy()[vclip])
    va_fr = sp == "val"
    Tf, calf = fit_temperature(Lc[va_fr], yc[va_fr])
    res = report(df, L[:, idx6], te, "face", T)
    res.update(val_clip_acc_subsampled=best, frame_temperature=Tf, frame_calibration=calf, calibration=cal,
               params=sum(p.numel() for p in m.parameters()), epochs=a.epochs, frames_per_clip=a.frames_per_clip,
               test_frame_acc=float((Lc[sp == "test"].argmax(1) == yc[sp == "test"]).mean()))
    np.save(Path(a.feats) / "face_mnv3_logits.npy", L); np.save(Path(a.feats) / "face_mnv3_frame_logits.npy", Lc)
    save("face_mnv3", res)
    log(f"face mnv3: TEST acc {res['test_acc']:.4f} (humans {res['human_acc_same_clips']:.4f})")


# ------------------------------------------------------------------------------------------------ fusion
def fuse_eval(a):
    from real_cremad import clip_fusion, fusion_study, tune_flag_thresholds, selective
    from emotion_edge.fusion.fuse import Thresholds
    from dataclasses import asdict
    df = C.metadata(a.root); df["split"] = C.actor_split(df)
    up = json.loads(OUT.read_text())
    Ls = np.load(Path(a.feats) / f"voice_{a.voice}_logits.npy"); Ts = up[f"voice_{a.voice}"]["temperature"]
    Lf = np.load(Path(a.feats) / f"face_{a.face}_logits.npy"); Tf = up[f"face_{a.face}"]["temperature"]
    A = np.load(Path(a.feats) / "audio.npz"); df["a_quality"] = A["quality"]
    Fz = np.load(Path(a.feats) / "faces112.npz"); clip = Fz["clip"]
    cnt = np.bincount(clip, minlength=len(df)); hasF = cnt > 0
    Fq = np.zeros(len(df)); np.add.at(Fq, clip, Fz["quality"]); Fq = np.where(hasF, Fq / np.maximum(cnt, 1), 0) * np.minimum(1, cnt / 4)
    okS = np.ones(len(df), bool)
    # temperatures were fitted on the 6-class sub-logits; the fusion uses full logits with the same temperature
    fused_fn = lambda th: clip_fusion(df, Ls, Ts, okS, Lf, Tf, hasF, Fq, th)
    res = {}
    fusion_study(df, fused_fn, res)
    th, f1 = tune_flag_thresholds(df, fused_fn, "val")
    res["thresholds_real_val"] = asdict(th)
    res["selective_test"] = selective(df, fused_fn, th, "test")
    save(f"fusion_{a.voice}_{a.face}", res)
    t = res["fusion_clip_level"]["test"]
    log(f"fused {a.voice}+{a.face}: TEST acc {t['fused_acc_vs_intended']:.4f} ECE {t['fused_ece_vs_intended']:.3f} "
        f"| answered {res['selective_test']['coverage_confident']:.2f} at {res['selective_test']['acc_confident']:.3f}")


# ------------------------------------------------------------------------------------------------ edge export
class VoiceEdge(nn.Module):
    """Raw 16 kHz waveform -> frozen SSL encoder -> per-layer mean/std pooling -> trained head -> 8 logits.
    One self-contained graph: the device needs no feature-extraction code."""

    def __init__(self, encoder, head):
        super().__init__()
        self.enc, self.head = encoder, head

    def forward(self, wav):                                    # wav: (1, samples) float32
        x = (wav - wav.mean(1, keepdim=True)) / (wav.std(1, keepdim=True) + 1e-7)
        hs = self.enc(x, output_hidden_states=True).hidden_states
        H = torch.stack(hs, 0)[:, 0]
        return self.head(torch.cat([H.mean(1), H.std(1)], -1)[None])


def export_voice(a):
    import librosa
    from transformers import AutoModel
    from emotion_edge.common import onnx_utils as ou
    up = json.loads(OUT.read_text())[f"voice_{a.tag}"]
    enc = AutoModel.from_pretrained(up["encoder"]).eval()
    Z = np.load(Path(a.feats) / f"ssl_{a.tag}.npz", allow_pickle=True)
    head = SSLHead(Z["X"].shape[1], Z["X"].shape[2]); head.load_state_dict(torch.load(Path(a.feats) / f"voice_{a.tag}_head.pt")); head.eval()
    model = VoiceEdge(enc, head).eval()
    out = Path(a.feats)
    ou.export_onnx(model, (torch.randn(1, 32000),), out / f"voice_{a.tag}_fp32.onnx", ["wav"], ["logits"], {"wav": {1: "samples"}}, opset=17)
    ou.quantize_dynamic_int8(out / f"voice_{a.tag}_fp32.onnx", out / f"voice_{a.tag}_int8.onnx")
    df = C.metadata(a.root); df["split"] = C.actor_split(df)
    te = np.where((df.split == "test").to_numpy())[0]
    idx6 = [SPEECH_LABELS.index(C.TO_SPEECH[c]) for c in C.CODES]
    res = {}
    for prec in ("fp32", "int8"):
        s = ou.ort_session(out / f"voice_{a.tag}_{prec}.onnx", 4)
        L, ms = [], []
        for i in te:
            y, _ = librosa.load(f"{a.root}/AudioWAV/{df.file[i]}.wav", sr=16000, duration=6.0)
            t = time.perf_counter(); L.append(s.run(None, {"wav": y[None].astype(np.float32)})[0][0]); ms.append((time.perf_counter() - t) * 1000)
        L = np.array(L)[:, idx6]
        res[prec] = {"test_acc": float((L.argmax(1) == df.y.to_numpy()[te]).mean()), "size_mb": ou.file_mb(out / f"voice_{a.tag}_{prec}.onnx"),
                     "ms_per_clip_4threads_p50": float(np.median(ms))}
        log(f"voice {a.tag} {prec}: acc {res[prec]['test_acc']:.4f} size {res[prec]['size_mb']:.1f} MB {res[prec]['ms_per_clip_4threads_p50']:.0f} ms/clip")
    save(f"voice_{a.tag}_edge", res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["voice", "face-extract", "face", "fuse", "export-voice"])
    ap.add_argument("--root", required=True); ap.add_argument("--feats", required=True)
    ap.add_argument("--ssl"); ap.add_argument("--tag")
    ap.add_argument("--epochs", type=int, default=8); ap.add_argument("--frames-per-clip", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--voice", default="distilhubert"); ap.add_argument("--face", default="mnv3")
    a = ap.parse_args()
    {"voice": train_voice, "face-extract": face_extract, "face": train_face, "fuse": fuse_eval, "export-voice": export_voice}[a.cmd](a)

