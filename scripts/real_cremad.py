"""Real-data study on CREMA-D: voice + face models, calibrated fusion vs human perception, and temporal retuning.

Requires the outputs of scripts/real_cremad_extract.py. Actor-independent splits (61 train / 15 val / 15 test actors):
every choice (epochs, temperature, thresholds, temporal parameters) is made on VAL actors; TEST actors are reported once.

    python scripts/real_cremad.py --root /path/to/cremad --feats /path/to/cremad_feats --out artifacts/cremad
"""
from __future__ import annotations
import argparse
import itertools
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scipy.special import softmax, log_softmax
from sklearn.metrics import f1_score, roc_auc_score

from emotion_edge.labels import CANON, SPEECH_LABELS, FACE_LABELS
from emotion_edge.realdata import cremad as C
from emotion_edge.common.metrics import fit_temperature, ece
from emotion_edge.common import onnx_utils as ou
from emotion_edge.speech.train import fit as fit_speech
from emotion_edge.vision.train import fit as fit_face
from emotion_edge.fusion.fuse import Modality, fuse, Thresholds, to_canonical
from emotion_edge.live.sources import Event
from emotion_edge.live.pipeline import LivePipeline
from emotion_edge.temporal.session import SessionConfig, DEFAULTS

CREMA_CANON = ["anger", "disgust", "fear", "joy", "neutral", "sadness"]


def crema6(p_canon):
    return C.canon_to_crema(p_canon)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------------------------------------ unimodal
def train_speech(df, A, out, res):
    X = A["X"].astype(np.float32)
    ok = ~np.isnan(X).any(1)
    y = C.speech_index(df.code)
    tr, va, te = [(df.split == s).to_numpy() & ok for s in ("train", "val", "test")]
    best = None
    for seed in range(3):                                   # 3 restarts, pick on val
        m, acc = fit_speech(X[tr], y[tr], X[va], y[va], epochs=150, seed=seed)
        if best is None or acc > best[1]:
            best = (m, acc, seed)
    m, vacc, seed = best
    with torch.no_grad():
        L = m(torch.from_numpy(np.nan_to_num(X))).numpy()
    T, cal = fit_temperature(L[va], y[va])
    ou.export_onnx(m, (torch.randn(1, X.shape[1]),), out / "speech_fp32.onnx", ["feat"], ["logits"], {"feat": {0: "b"}, "logits": {0: "b"}})
    ou.quantize_dynamic_int8(out / "speech_fp32.onnx", out / "speech_int8.onnx")
    L8 = ou.ort_session(out / "speech_int8.onnx", 4).run(None, {"feat": np.nan_to_num(X)})[0]
    res["speech"] = summarise_unimodal(df, L, L8, y, te, T, cal, "voice", SPEECH_LABELS, vacc,
                                       params=sum(p.numel() for p in m.parameters()), size_mb=ou.file_mb(out / "speech_int8.onnx"))
    res["speech"]["restart_seed"] = seed
    np.save(out / "speech_logits.npy", L)
    return L, T, ok


def train_face(df, F, out, res, epochs):
    crops, clip = F["crops"], F["clip"]
    split_of_crop = df.split.to_numpy()[clip]
    yc = C.face_index(df.code)[clip]
    tr, va = split_of_crop == "train", split_of_crop == "val"
    log(f"face crops: train {tr.sum()} val {va.sum()} test {(split_of_crop == 'test').sum()}")
    m, vacc = fit_face(crops[tr], yc[tr], crops[va], yc[va], epochs=epochs)
    f = lambda a: torch.from_numpy(a).float()[:, None] / 255
    with torch.no_grad():
        Lc = np.concatenate([m(f(crops[i:i + 512])).numpy() for i in range(0, len(crops), 512)])
    # clip-level logits = mean of per-frame log-probabilities (product of frame experts, geometric mean)
    n = len(df)
    Lclip = np.zeros((n, len(FACE_LABELS)))
    cnt = np.bincount(clip, minlength=n)
    np.add.at(Lclip, clip, log_softmax(Lc, 1))
    has = cnt > 0
    Lclip[has] /= cnt[has, None]
    y = C.face_index(df.code)
    vclip = (df.split == "val").to_numpy() & has
    T, cal = fit_temperature(Lclip[vclip], y[vclip])                 # clip-level calibration
    Tf, calf = fit_temperature(Lc[va], yc[va])                       # frame-level calibration (for the live filter)
    ex = torch.rand(1, 1, 48, 48)
    ou.export_onnx(m, (ex,), out / "face_fp32.onnx", ["img"], ["logits"], {"img": {0: "b"}, "logits": {0: "b"}})
    rng = np.random.default_rng(0)
    calib = [f(crops[tr][i:i + 1]).numpy() for i in rng.choice(tr.sum(), 300, replace=False)]
    ou.quantize_static_int8(out / "face_fp32.onnx", out / "face_int8.onnx", calib, "img")
    s8 = ou.ort_session(out / "face_int8.onnx", 4)
    Lc8 = np.concatenate([s8.run(None, {"img": f(crops[i:i + 256]).numpy()})[0] for i in range(0, len(crops), 256)])
    L8 = np.zeros_like(Lclip); np.add.at(L8, clip, log_softmax(Lc8, 1)); L8[has] /= cnt[has, None]
    te = (df.split == "test").to_numpy() & has
    res["face"] = summarise_unimodal(df, Lclip, L8, y, te, T, cal, "face", FACE_LABELS, vacc,
                                     params=sum(p.numel() for p in m.parameters()), size_mb=ou.file_mb(out / "face_int8.onnx"))
    res["face"].update({"frame_temperature": Tf, "frame_calibration": calf,
                        "test_frame_acc": float((Lc[split_of_crop == "test"].argmax(1) == yc[split_of_crop == "test"]).mean()),
                        "clips_without_face": int((~has).sum()), "frames_per_clip_mean": float(cnt.mean())})
    np.save(out / "face_frame_logits.npy", Lc)
    return Lclip, Lc, T, Tf, has


def summarise_unimodal(df, L, L8, y, te, T, cal, human_mod, labels, vacc, **kw):
    """Accuracy over the 6 CREMA-D classes vs the intended label, vs the human majority for this modality,
    and the humans' own accuracy on the same clips for reference."""
    idx = [labels.index(n) for n in (C.TO_SPEECH if labels is SPEECH_LABELS else C.TO_FACE).values()]  # A D F H N S order
    p6 = softmax(L[:, idx] / T, 1)
    p6_8 = softmax(L8[:, idx], 1)
    yi = df.y.to_numpy()
    pred = p6.argmax(1)
    hum = df[f"h_{human_mod}_top"].to_numpy()
    return {"val_acc": vacc, "temperature": T, **{f"cal_{k}": v for k, v in cal.items()},
            "test_acc_vs_intended": float((pred[te] == yi[te]).mean()),
            "test_macro_f1": float(f1_score(yi[te], pred[te], average="macro")),
            "test_int8_acc_vs_intended": float((p6_8.argmax(1)[te] == yi[te]).mean()),
            "int8_agreement": float((p6_8.argmax(1)[te] == pred[te]).mean()),
            "test_acc_vs_human_majority": float((pred[te] == hum[te]).mean()),
            "human_acc_vs_intended_same_clips": float((hum[te] == yi[te]).mean()),
            "per_class_recall": {CREMA_CANON[k]: float((pred[te][yi[te] == k] == k).mean()) for k in range(6)},
            "confusion": np.bincount(yi[te] * 6 + pred[te], minlength=36).reshape(6, 6).tolist(), **kw}


# ------------------------------------------------------------------------------------------------ fusion
def clip_fusion(df, Ls, Ts, okS, Lf, Tf, hasF, Fq, th=Thresholds()):
    out = []
    for i in range(len(df)):
        mods = []
        if okS[i]:
            mods.append(Modality("speech", logits=Ls[i], temperature=Ts, quality=float(df.a_quality.iloc[i])))
        if hasF[i]:
            mods.append(Modality("face", logits=Lf[i], temperature=Tf, quality=float(Fq[i])))
        out.append(fuse(mods, th=th) if mods else None)
    return out


def fusion_study(df, fused_fn, res):
    yi = df.y.to_numpy()
    out = {}
    for split in ("val", "test"):
        m = (df.split == split).to_numpy()
        fz = fused_fn(Thresholds())
        idx = np.where(m & np.array([f is not None for f in fz]))[0]
        P = np.stack([crema6(fz[i].posterior)[0] for i in idx])
        pred = P.argmax(1)
        hav = np.stack(df.h_av.to_numpy())[idx]
        amb = df.h_av_agree.to_numpy()[idx] < 0.6
        hv, hf = df.h_voice_top.to_numpy()[idx], df.h_face_top.to_numpy()[idx]
        conf_v, conf_f = df.h_voice_agree.to_numpy()[idx] >= 0.6, df.h_face_agree.to_numpy()[idx] >= 0.6
        incong = (hv != hf) & conf_v & conf_f                       # raters were confident in each modality and disagreed
        p1 = np.array([fz[i].p_top1 for i in idx]); H = np.array([fz[i].entropy for i in idx]); cf = np.array([fz[i].conflict for i in idx])
        out[split] = {"n": int(len(idx)), "fused_acc_vs_intended": float((pred == yi[idx]).mean()),
                      "fused_macro_f1": float(f1_score(yi[idx], pred, average="macro")),
                      "fused_acc_vs_human_av_majority": float((pred == hav.argmax(1)).mean()),
                      "human_av_acc_vs_intended": float((hav.argmax(1) == yi[idx]).mean()),
                      "fused_ece_vs_intended": ece(P, yi[idx]),
                      "brier_vs_human_av_votes": float(((P - hav) ** 2).sum(1).mean()),
                      "brier_onehot_intended_vs_human_av_votes": float(((np.eye(6)[yi[idx]] - hav) ** 2).sum(1).mean()),
                      "human_ambiguous_rate": float(amb.mean()), "human_incongruent_rate": float(incong.mean()),
                      "auroc_1_minus_p_top1__human_ambiguous": float(roc_auc_score(amb, 1 - p1)),
                      "auroc_entropy__human_ambiguous": float(roc_auc_score(amb, H)),
                      "auroc_conflict__human_incongruent": float(roc_auc_score(incong, cf)) if incong.any() else None,
                      "auroc_1_minus_p_top1__fused_wrong": float(roc_auc_score(pred != yi[idx], 1 - p1))}
    res["fusion_clip_level"] = out


def tune_flag_thresholds(df, fused_fn, split="val"):
    """Thresholds tuned on REAL val actors: flag = not 'confident'; should_flag = fused wrong OR humans ambiguous."""
    grid = {"conf_p": [0.4, 0.5, 0.6, 0.7], "margin": [0.1, 0.2, 0.3], "entropy": [0.6, 0.7, 0.8, 0.9], "conflict_jsd": [0.2, 0.3, 0.4, 0.5]}
    m = (df.split == split).to_numpy()
    yi = df.y.to_numpy(); amb = df.h_av_agree.to_numpy() < 0.6
    best = (-1, None, None)
    for vals in itertools.product(*grid.values()):
        th = Thresholds(**dict(zip(grid, vals)))
        fz = fused_fn(th)
        tp = fp = fn = 0
        for i in np.where(m)[0]:
            if fz[i] is None:
                continue
            wrong = crema6(fz[i].posterior)[0].argmax() != yi[i]
            should, flag = wrong or amb[i], fz[i].state != "confident"
            tp += should and flag; fp += flag and not should; fn += should and not flag
        f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        if f1 > best[0]:
            best = (f1, th, None)
    return best[1], best[0]


def selective(df, fused_fn, th, split):
    m = (df.split == split).to_numpy()
    fz = fused_fn(th)
    yi = df.y.to_numpy()
    idx = [i for i in np.where(m)[0] if fz[i] is not None]
    ok = np.array([crema6(fz[i].posterior)[0].argmax() == yi[i] for i in idx])
    conf = np.array([fz[i].state == "confident" for i in idx])
    states = np.array([fz[i].state for i in idx])
    amb = df.h_av_agree.to_numpy()[idx] < 0.6
    return {"coverage_confident": float(conf.mean()), "acc_confident": float(ok[conf].mean()), "acc_flagged": float(ok[~conf].mean()),
            "acc_all": float(ok.mean()), "human_ambiguous_rate_confident": float(amb[conf].mean()),
            "human_ambiguous_rate_flagged": float(amb[~conf].mean()),
            "state_share": {s: float((states == s).mean()) for s in np.unique(states)}}


# ------------------------------------------------------------------------------------------------ temporal
def build_sessions(df, split, A, F, Ls, Lc, rng, gap=(0.3, 1.2), ep=(3, 6)):
    """Per actor: real clips arranged into emotion episodes (3-6 clips of one acted emotion, then another).
    Events = one speech block per clip (real prosody-model output) + face blocks at the real sampled frame times."""
    sessions = []
    clip_of_crop, crop_time, crop_q = F["clip"], F["time"], F["quality"]
    crops_by_clip = {}
    for k, c in enumerate(clip_of_crop):
        crops_by_clip.setdefault(int(c), []).append(k)
    for actor, g in df[df.split == split].groupby("actor"):
        pools = {c: list(rng.permutation(g.index[g.code == c].to_numpy())) for c in C.CODES}
        t, ev, truth, prev = 0.0, [], [], None
        while any(pools.values()):
            choices = [c for c in C.CODES if pools[c] and c != prev] or [c for c in C.CODES if pools[c]]
            c = rng.choice(choices)
            for _ in range(min(int(rng.integers(ep[0], ep[1] + 1)), len(pools[c]))):
                i = int(pools[c].pop())
                dur = float(A["duration"][i]) or 2.5
                for k in crops_by_clip.get(i, []):
                    ev.append(Event(t + float(crop_time[k]), "block", {"modality": "face", "logits": Lc[k], "quality": float(crop_q[k]), "dur": 0.25}))
                if not np.isnan(A["X"][i]).any():
                    ev.append(Event(t + dur, "block", {"modality": "speech", "logits": Ls[i], "quality": float(A["quality"][i]), "dur": dur}))
                truth.append((t, t + dur, C.NAMES[C.CODES.index(c)]))
                t += dur + float(rng.uniform(*gap))
            prev = c
        sessions.append({"actor": int(actor), "events": sorted(ev), "truth": truth, "duration": t})
    return sessions


def truth_at(truth, t):
    for a, b, l in truth:
        if a <= t < b:
            return l
    return None


def eval_sessions(sessions, cfg_kwargs, temps, window=20.0):
    accs, flick, lat, mood_ok, mood_p, n_inst = [], [], [], [], [], 0
    for s in sessions:
        cfg = SessionConfig(window_s=window, hop_s=5.0, tick_s=0.5, temperatures=dict(temps), modality=cfg_kwargs)
        recs = LivePipeline(cfg).run(s["events"])
        inst = [r for r in recs if r["type"] == "instant" and r.get("label")]
        lab = {"love": "joy"}
        seq = [(r["t"], lab.get(r["label"], r["label"]), truth_at(s["truth"], r["t"])) for r in inst]
        seq = [x for x in seq if x[2] is not None]
        accs += [p == y for _, p, y in seq]
        labels = [p for _, p, _ in seq]
        flick.append(sum(a != b for a, b in zip(labels[1:], labels[:-1])) / (s["duration"] / 60))
        # latency at episode boundaries
        bounds = [(a, l) for (a, _, l), (_, _, pl) in zip(s["truth"][1:], s["truth"][:-1]) if l != pl]
        for b, l in bounds:
            after = [(t, p) for t, p, _ in seq if t >= b]
            hit = next((t for k, (t, p) in enumerate(after) if all(q == l for _, q in after[k:k + 2])), None)
            if hit is not None and hit - b < 15:
                lat.append(hit - b)
        for r in recs:
            if r["type"] != "window" or not r.get("fused_mood"):
                continue
            ts = np.arange(max(0, r["t"] - window), r["t"], 0.5)
            tl = [truth_at(s["truth"], x) for x in ts]
            tl = [x for x in tl if x]
            if len(tl) < 10:
                continue
            maj = max(set(tl), key=tl.count)
            if tl.count(maj) / len(tl) < 0.7:
                continue
            mood_ok.append(lab.get(r["fused_mood"]["label"], r["fused_mood"]["label"]) == maj)
            mood_p.append(r["fused_mood"]["p_dominant"])
    mood_ok, mood_p = np.array(mood_ok), np.array(mood_p)
    rel = []
    for lo, hi in zip(np.linspace(0, 1, 6)[:-1], np.linspace(0, 1, 6)[1:]):
        sel = (mood_p >= lo) & (mood_p < hi + (hi == 1))
        if sel.any():
            rel.append({"bin": [round(lo, 1), round(hi, 1)], "n": int(sel.sum()), "mean_p": float(mood_p[sel].mean()), "acc": float(mood_ok[sel].mean())})
    return {"instant_acc": float(np.mean(accs)), "flicker_per_min": float(np.mean(flick)), "latency_s": float(np.median(lat)) if lat else None,
            "mood_acc": float(mood_ok.mean()) if len(mood_ok) else None, "n_mood_windows": int(len(mood_ok)),
            "mood_p_dominant_ece": float(sum(b["n"] * abs(b["acc"] - b["mean_p"]) for b in rel) / max(1, len(mood_ok))), "mood_reliability": rel}


def static_baseline(sessions, temps, horizon=2.0):
    accs, flick = [], []
    for s in sessions:
        last, i, labels = {}, 0, []
        ev = s["events"]
        for t in np.arange(0, s["duration"], 0.5):
            while i < len(ev) and ev[i].t <= t:
                last[ev[i].data["modality"]] = (ev[i].t, ev[i].data); i += 1
            mods = [Modality(m, logits=d["logits"], temperature=temps[m], quality=d["quality"]) for m, (tt, d) in last.items() if t - tt <= horizon]
            y = truth_at(s["truth"], t)
            if not mods or y is None:
                continue
            l = fuse(mods).label
            l = "joy" if l == "love" else l
            accs.append(l == y); labels.append(l)
        flick.append(sum(a != b for a, b in zip(labels[1:], labels[:-1])) / (s["duration"] / 60))
    return {"instant_acc": float(np.mean(accs)), "flicker_per_min": float(np.mean(flick))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--feats", required=True)
    ap.add_argument("--out", default="artifacts/cremad")
    ap.add_argument("--face-epochs", type=int, default=25)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    df = C.metadata(a.root)
    df["split"] = C.actor_split(df)
    A = dict(np.load(Path(a.feats) / "audio.npz"))
    F = dict(np.load(Path(a.feats) / "faces.npz"))
    assert list(A["files"]) == list(df.file) and list(F["files"]) == list(df.file)
    df["a_quality"] = A["quality"]
    res = {"dataset": "CREMA-D", "split": {s: {"clips": int((df.split == s).sum()), "actors": int(df[df.split == s].actor.nunique())} for s in ("train", "val", "test")},
           "human_reference_all_clips": {m: float((df[f"h_{m}_top"] == df.y).mean()) for m in ("voice", "face", "av")}}

    log("training speech"); Ls, Ts, okS = train_speech(df, A, out, res); log("speech", {k: res["speech"][k] for k in ("test_acc_vs_intended", "human_acc_vs_intended_same_clips")})
    log("training face"); Lf, Lc, Tf_clip, Tf_frame, hasF = train_face(df, F, out, res, a.face_epochs); log("face", {k: res["face"][k] for k in ("test_acc_vs_intended", "human_acc_vs_intended_same_clips")})
    cnt = np.bincount(F["clip"], minlength=len(df))
    Fq = np.zeros(len(df)); np.add.at(Fq, F["clip"], F["quality"]); Fq = np.where(cnt > 0, Fq / np.maximum(cnt, 1), 0) * np.minimum(1, cnt / 4)
    fused_fn = lambda th: clip_fusion(df, Ls, Ts, okS, Lf, Tf_clip, hasF, Fq, th)

    log("fusion study"); fusion_study(df, fused_fn, res)
    th_sim = Thresholds(**json.loads(Path("results/fusion_sim.json").read_text())["tuned_thresholds"]) if Path("results/fusion_sim.json").exists() else Thresholds()
    th_real, f1 = tune_flag_thresholds(df, fused_fn, "val")
    res["thresholds"] = {"tuned_on_real_val": asdict(th_real), "val_flag_f1": f1, "from_simulation": asdict(th_sim)}
    res["selective_test"] = {"real_tuned": selective(df, fused_fn, th_real, "test"), "simulation_tuned": selective(df, fused_fn, th_sim, "test"),
                             "default": selective(df, fused_fn, Thresholds(), "test")}
    log("thresholds", res["thresholds"]["tuned_on_real_val"])

    log("temporal retuning")
    rng = np.random.default_rng(0)
    val_s = build_sessions(df, "val", A, F, Ls, Lc, rng)
    test_s = build_sessions(df, "test", A, F, Ls, Lc, np.random.default_rng(1))
    temps = {"speech": Ts, "face": Tf_frame, "text": 1.0}
    sweep = []
    for fd, sd, tc in itertools.product([1.0, 2.0, 3.0, 5.0, 8.0], [2.0, 6.0, 10.0], [0.5, 1.0]):
        mk = {"face": {"dwell_s": fd, "tau_c": tc, "base_weight": 1.0}, "speech": {"dwell_s": sd, "tau_c": 2.0, "base_weight": 1.0},
              "text": dict(DEFAULTS["text"])}
        r = eval_sessions(val_s, mk, temps)
        sweep.append({"face_dwell": fd, "speech_dwell": sd, "face_tau_c": tc, **{k: r[k] for k in ("instant_acc", "flicker_per_min", "mood_acc", "mood_p_dominant_ece")}})
    # select: best instant accuracy; ties (within 0.5 pt) broken by lower flicker
    top = max(s["instant_acc"] for s in sweep)
    cand = [s for s in sweep if s["instant_acc"] >= top - 0.005]
    best = min(cand, key=lambda s: s["flicker_per_min"])
    chosen = {"face": {"dwell_s": best["face_dwell"], "tau_c": best["face_tau_c"], "base_weight": 1.0},
              "speech": {"dwell_s": best["speech_dwell"], "tau_c": 2.0, "base_weight": 1.0}, "text": dict(DEFAULTS["text"])}
    windows = {}
    for w in (10.0, 20.0, 30.0):
        windows[str(w)] = {k: v for k, v in eval_sessions(val_s, chosen, temps, window=w).items() if k != "mood_reliability"}
    res["temporal"] = {
        "sessions": {"val": len(val_s), "test": len(test_s), "mean_duration_s": float(np.mean([s["duration"] for s in test_s]))},
        "val_sweep": sweep, "chosen_on_val": chosen, "val_window_sweep": windows,
        "test_default_params": eval_sessions(test_s, {k: dict(v) for k, v in DEFAULTS.items()}, temps),
        "test_tuned_params": eval_sessions(test_s, chosen, temps),
        "test_static_baseline": static_baseline(test_s, temps)}
    Path("results").mkdir(exist_ok=True)
    Path("results/real_cremad.json").write_text(json.dumps(res, indent=2, default=float))
    log("done ->", "results/real_cremad.json")
    print(json.dumps({k: res[k] for k in ("human_reference_all_clips", "fusion_clip_level", "selective_test")}, indent=1, default=float))
    print(json.dumps({k: v for k, v in res["temporal"].items() if k not in ("val_sweep",)}, indent=1, default=float))


if __name__ == "__main__":
    main()
