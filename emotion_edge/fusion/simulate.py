"""Synthetic modality-output simulator used to *validate the fusion logic* (not to claim real accuracy).

Each sample has a true canonical emotion. Every modality emits logits = strength * onehot(own label) + noise.
Scenario mix:
  clean       all present modalities see the true emotion
  dropout     a modality is missing (no face / silence / no transcript)
  blend       true state is a mixture of two V/A-compatible emotions; modalities each see one of them
  incongruent one modality sees a *different, incompatible* emotion (sarcasm / masking / sensor failure)
Strengths are assumptions chosen to resemble lightweight real models (text ~93%, speech ~70%, face ~65% on 6-8 classes).
"""
from __future__ import annotations
import numpy as np
from emotion_edge.labels import CANON, LABELS, MAPS, CIDX, coverage
from emotion_edge.fusion.fuse import Modality, fuse, Thresholds, _VA_DIST

STRENGTH = {"text": 4.0, "speech": 2.2, "face": 2.0}
TEMP = {"text": 1.0, "speech": 1.0, "face": 1.0}
BLENDS = [("joy", "love"), ("anger", "disgust"), ("fear", "surprise"), ("fear", "sadness")]


def _canon_to_label(canon: str, modality: str, rng):
    """Pick a modality label that maps to `canon` (None if uncovered)."""
    M = MAPS[modality]
    cands = np.where(M[:, CIDX[canon]] > 0)[0]
    return None if len(cands) == 0 else int(rng.choice(cands))


def _logits(modality, canon, rng, strength_scale=1.0):
    n = len(LABELS[modality])
    z = rng.normal(0, 1, n)
    lab = _canon_to_label(canon, modality, rng)
    if lab is not None:
        z[lab] += STRENGTH[modality] * strength_scale
    return z


def make_dataset(n=4000, seed=0, p_dropout=0.25, p_blend=0.12, p_incong=0.15):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        u = rng.random()
        true = CANON[rng.integers(len(CANON))]
        kind = "clean"
        seen = {m: true for m in LABELS}
        if u < p_incong:
            kind = "incongruent"
            m = rng.choice(["speech", "face"])
            far = sorted(CANON, key=lambda c: -_VA_DIST[CIDX[c], CIDX[true]])[:3]  # three most V/A-distant emotions
            seen[m] = rng.choice(far)
        elif u < p_incong + p_blend:
            kind = "blend"
            a, b = BLENDS[rng.integers(len(BLENDS))]
            true = a
            for m in LABELS:
                seen[m] = a if rng.random() < 0.5 else b
        present = {m: True for m in LABELS}
        if rng.random() < p_dropout and kind == "clean":
            kind = "dropout"
            present[rng.choice(list(LABELS))] = False
        mods = {}
        for m in LABELS:
            if present[m]:
                q = float(np.clip(rng.beta(6, 2), 0.2, 1))        # per-input quality
                mods[m] = Modality(m, logits=_logits(m, seen[m], rng, 0.6 + 0.4 * q), temperature=TEMP[m], quality=q)
        rows.append({"true": true, "kind": kind, "mods": mods, "seen": seen})
    return rows


def evaluate(rows, th=Thresholds()):
    out = {"n": len(rows)}
    uni = {m: [] for m in LABELS}
    fused_ok, states, kinds, pt1, conf = [], [], [], [], []
    for r in rows:
        f = fuse(list(r["mods"].values()), th=th)
        ok = f.label == r["true"] or (r["kind"] == "blend" and r["true"] in f.top2)
        fused_ok.append(ok); states.append(f.state); kinds.append(r["kind"]); pt1.append(f.p_top1); conf.append(f.conflict)
        for m, mod in r["mods"].items():
            from emotion_edge.fusion.fuse import to_canonical, _norm_probs
            c = CANON[int(to_canonical(_norm_probs(mod), m).argmax())]
            uni[m].append((c == r["true"], r["kind"]))
    fused_ok, states, kinds = np.array(fused_ok), np.array(states), np.array(kinds)
    clean = np.isin(kinds, ["clean", "dropout"])
    out["unimodal_acc_clean"] = {m: float(np.mean([a for a, k in v if k in ("clean", "dropout")])) for m, v in uni.items()}
    out["fused_acc_clean"] = float(fused_ok[clean].mean())
    out["fused_acc_all"] = float(fused_ok.mean())
    out["state_by_kind"] = {k: {s: float(np.mean(states[kinds == k] == s)) for s in
                                ["confident", "blend", "ambiguous", "conflict", "uncertain"]} for k in np.unique(kinds)}
    flagged = states != "confident"
    out["flag_rate_by_kind"] = {k: float(flagged[kinds == k].mean()) for k in np.unique(kinds)}
    out["acc_when_confident"] = float(fused_ok[~flagged].mean()) if (~flagged).any() else None
    out["acc_when_flagged"] = float(fused_ok[flagged].mean()) if flagged.any() else None
    out["coverage_confident"] = float((~flagged).mean())
    out["_arrays"] = {"correct": fused_ok, "score": np.array(pt1) - np.array(conf), "kinds": kinds, "states": states}
    return out


def tune_thresholds(rows_val, grid=None):
    """Grid-search ambiguity thresholds on a validation set.
    Objective: F1 of the decision 'flag this prediction for review', where 'should flag' = the fused label is wrong
    OR the sample is an incongruent/blend case. Real deployments should swap `should_flag` for human-labelled
    ambiguity annotations; the machinery is unchanged."""
    import itertools
    grid = grid or {"conf_p": [0.45, 0.55, 0.65], "margin": [0.1, 0.2, 0.3], "entropy": [0.7, 0.8, 0.9],
                    "conflict_jsd": [0.2, 0.3, 0.4, 0.5]}
    best = (-1, None)
    for vals in itertools.product(*grid.values()):
        th = Thresholds(**dict(zip(grid, vals)))
        tp = fp = fn = 0
        for r in rows_val:
            f = fuse(list(r["mods"].values()), th=th)
            wrong = not (f.label == r["true"] or (r["kind"] == "blend" and r["true"] in f.top2))
            should = wrong or r["kind"] in ("incongruent", "blend")
            flag = f.state != "confident"
            tp += flag and should; fp += flag and not should; fn += (not flag) and should
        f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        if f1 > best[0]:
            best = (f1, th)
    return best[1], best[0]
