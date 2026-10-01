"""Face model upgrade with an EXPRESSION-pretrained encoder: HSEmotion EfficientNet-B0 trained on AffectNet
(Savchenko, 2022; code Apache-2.0; weights trained on AffectNet, whose licence is research-only).

The published ONNX graph (no pickle) is used as a frozen encoder: its 1280-d global-pool output is exposed as an extra
output. Two evaluations on the same CREMA-D actor split as before:
  zero-shot : AffectNet's own 8-way head mapped to the 6 CREMA-D classes (no CREMA-D training at all)
  head      : logistic-regression / MLP head on frame embeddings, trained on train actors, chosen on val actors

    python scripts/face_hse.py --root <cremad> --feats <feats> --onnx /path/enet_b0_8_best_afew.onnx
"""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scipy.special import softmax, log_softmax
from sklearn.metrics import f1_score
from emotion_edge.realdata import cremad as C
from emotion_edge.common.metrics import fit_temperature
from emotion_edge.labels import FACE_LABELS

AFFECTNET = ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
TO_CREMA = {"A": "anger", "D": "disgust", "F": "fear", "H": "happiness", "N": "neutral", "S": "sadness"}
MEAN, STD = np.array([0.485, 0.456, 0.406])[:, None, None], np.array([0.229, 0.224, 0.225])[:, None, None]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def encoder_with_features(path, out):
    import onnx
    m = onnx.load(path)
    feat = next(n.output[0] for n in m.graph.node if n.op_type == "Flatten")
    m.graph.output.append(onnx.helper.make_tensor_value_info(feat, onnx.TensorProto.FLOAT, ["batch_size", 1280]))
    onnx.save(m, out)
    return out


def prep(crops112):
    import cv2
    x = np.stack([cv2.resize(c, (224, 224), interpolation=cv2.INTER_LINEAR) for c in crops112]).astype(np.float32) / 255
    x = np.repeat(x[:, None], 3, 1)                                   # grayscale -> 3 channels
    return ((x - MEAN) / STD).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--feats", required=True); ap.add_argument("--onnx", required=True)
    ap.add_argument("--max-frames", type=int, default=6)
    a = ap.parse_args()
    import onnxruntime as ort
    feats_dir = Path(a.feats)
    df = C.metadata(a.root); df["split"] = C.actor_split(df)
    Fz = np.load(feats_dir / "faces112.npz")
    crops, clip = Fz["crops"], Fz["clip"]
    # up to max_frames evenly spaced frames per clip
    keep = []
    order = np.argsort(clip, kind="stable")
    starts = np.searchsorted(clip[order], np.arange(len(df)))
    ends = np.searchsorted(clip[order], np.arange(len(df)), side="right")
    for c in range(len(df)):
        ids = order[starts[c]:ends[c]]
        if len(ids):
            keep.extend(ids[np.linspace(0, len(ids) - 1, min(a.max_frames, len(ids))).astype(int)])
    keep = np.array(sorted(set(keep)))
    cache = feats_dir / "hse_frame_feats.npz"
    if cache.exists():
        Z = np.load(cache); E, Lz = Z["E"], Z["L"]; keep = Z["keep"]
    else:
        so = ort.SessionOptions(); so.intra_op_num_threads = 4
        sess = ort.InferenceSession(encoder_with_features(a.onnx, str(feats_dir / "hse_feat.onnx")), so, providers=["CPUExecutionProvider"])
        E, Lz = np.zeros((len(keep), 1280), np.float32), np.zeros((len(keep), 8), np.float32)
        t0 = time.time()
        for i in range(0, len(keep), 64):
            b = keep[i:i + 64]
            lo, fe = sess.run(None, {"input": prep(crops[b])})
            E[i:i + 64], Lz[i:i + 64] = fe, lo
            if i % 6400 == 0:
                log(f"{i}/{len(keep)} frames {time.time() - t0:.0f}s")
        np.savez(cache, E=E, L=Lz, keep=keep)
    fclip = clip[keep]
    y = df.y.to_numpy()
    sp = df.split.to_numpy()
    res = {"frames_used": int(len(keep))}

    def clip_pool(frame_logits):
        n = len(df); L = np.zeros((n, frame_logits.shape[1])); cnt = np.bincount(fclip, minlength=n)
        np.add.at(L, fclip, log_softmax(frame_logits, 1)); L[cnt > 0] /= cnt[cnt > 0, None]
        return L, cnt > 0

    # zero-shot
    idx6 = [AFFECTNET.index(TO_CREMA[c]) for c in C.CODES]
    Lzs, has = clip_pool(Lz)
    te = (sp == "test") & has
    pz = Lzs[:, idx6].argmax(1)
    res["zero_shot"] = {"test_acc": float((pz[te] == y[te]).mean()), "test_macro_f1": float(f1_score(y[te], pz[te], average="macro")),
                        "per_class_recall": {C.NAMES[k]: float((pz[te][y[te] == k] == k).mean()) for k in range(6)}}
    log("zero-shot test acc", res["zero_shot"]["test_acc"])

    # trained head on frame embeddings (frame-level so it also serves the live 4 fps path)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    yf = C.face_index(df.code)[fclip]
    trf, vaf = sp[fclip] == "train", sp[fclip] == "val"
    sc = StandardScaler().fit(E[trf])
    Etr, Eva, Eall = sc.transform(E[trf]), sc.transform(E[vaf]), sc.transform(E)
    best = None
    vclip = (sp == "val") & has
    for Cc in (0.01, 0.03, 0.1, 0.3, 1.0):
        clf = LogisticRegression(C=Cc, max_iter=3000).fit(Etr, yf[trf])
        Lf = np.full((len(E), len(FACE_LABELS)), -30.0)
        Lf[:, clf.classes_] = clf.decision_function(Eall)
        Lc, _ = clip_pool(Lf)
        acc = float((Lc[vclip][:, [FACE_LABELS.index(C.TO_FACE[c]) for c in C.CODES]].argmax(1) == y[vclip]).mean())
        log(f"head C={Cc}: val clip acc {acc:.4f}")
        if best is None or acc > best[0]:
            best = (acc, Cc, clf, Lf, Lc)
    vacc, Cc, clf, Lf, Lc = best
    i6 = [FACE_LABELS.index(C.TO_FACE[c]) for c in C.CODES]
    T, cal = fit_temperature(Lc[vclip][:, i6], y[vclip])
    Tf, calf = fit_temperature(Lf[vaf][:, clf.classes_], np.searchsorted(clf.classes_, yf[vaf]))
    p = Lc[:, i6].argmax(1)
    hum = df.h_face_top.to_numpy()
    res["head"] = {"C": Cc, "val_clip_acc": vacc, "test_acc": float((p[te] == y[te]).mean()),
                   "test_macro_f1": float(f1_score(y[te], p[te], average="macro")),
                   "human_acc_same_clips": float((hum[te] == y[te]).mean()), "temperature": T, "frame_temperature": Tf,
                   "calibration": cal, "per_class_recall": {C.NAMES[k]: float((p[te][y[te] == k] == k).mean()) for k in range(6)},
                   "confusion": np.bincount(y[te] * 6 + p[te], minlength=36).reshape(6, 6).tolist(),
                   "encoder": "HSEmotion enet_b0_8_best_afew (EfficientNet-B0, AffectNet)", "encoder_params": 3996789}
    np.save(feats_dir / "face_hse_logits.npy", Lc); np.save(feats_dir / "face_hse_frame_logits.npy", Lf)
    import pickle
    pickle.dump({"scaler": sc, "clf": clf}, open(feats_dir / "face_hse_head.pkl", "wb"))
    out = Path("results/upgrade.json"); r = json.loads(out.read_text()) if out.exists() else {}
    r["face_hse"] = res; out.write_text(json.dumps(r, indent=2, default=float))
    log(f"HSEmotion head: TEST acc {res['head']['test_acc']:.4f} (humans {res['head']['human_acc_same_clips']:.4f}); zero-shot {res['zero_shot']['test_acc']:.4f}")


if __name__ == "__main__":
    main()
