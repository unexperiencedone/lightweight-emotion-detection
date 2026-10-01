"""Quantitative check of the temporal layer on the scripted synthetic session (emotion_edge/live/scenario.py).

Baseline = what you get WITHOUT the temporal layer: at each tick, fuse the latest block of every modality seen in the
last 2 s (static fusion only). Compared with the sticky-filter instant emotion on:
  accuracy in clean phases, flicker (label switches/min), transition latency, conflict detection (masking phase),
  false-alarm rate of 'conflict' in clean phases, escalation tagging, and window-mood accuracy.
Everything here is SYNTHETIC: it measures the logic of the temporal layer, not real-world performance.
"""
from __future__ import annotations
import json
import numpy as np

from emotion_edge.fusion.fuse import Modality, fuse
from emotion_edge.live import scenario as S
from emotion_edge.live.pipeline import LivePipeline

CLEAN = ("neutral", "joy", "sadness")
BOUNDARIES = [(20.0, "joy"), (62.0, "sadness")]


def baseline_instants(events, tick_s=0.5, horizon=2.0):
    last, out, i = {}, [], 0
    ev = sorted(events)
    for t in np.arange(0, S.DURATION + tick_s, tick_s):
        while i < len(ev) and ev[i].t <= t:
            d = ev[i].data
            last[d["modality"]] = (ev[i].t, d)
            i += 1
        mods = [Modality(m, logits=d["logits"], quality=d["quality"]) for m, (tt, d) in last.items() if t - tt <= horizon]
        if mods:
            f = fuse(mods)
            out.append({"t": float(t), "label": f.label, "state": f.state})
    return out


def _switches_per_min(labels, times):
    if len(labels) < 2:
        return 0.0
    return float((np.array(labels[1:]) != np.array(labels[:-1])).sum() / ((times[-1] - times[0]) / 60))


def _latency(inst, boundary, new_label, hold_s=2.0, tick_s=0.5):
    need = int(hold_s / tick_s)
    seq = [(r["t"], r["label"]) for r in inst if r["t"] >= boundary]
    for k in range(len(seq) - need):
        if all(l == new_label for _, l in seq[k:k + need]):
            return seq[k][0] - boundary
    return float("nan")


def evaluate_seed(seed: int) -> dict:
    events = S.events(seed)
    recs = LivePipeline().run(events)
    inst = [r for r in recs if r["type"] == "instant" and r.get("label")]
    base = baseline_instants(events)
    out = {}
    for name, seq in (("temporal", inst), ("baseline", base)):
        phase = np.array([S.truth_at(r["t"])[0] for r in seq])
        truth = np.array([S.truth_at(r["t"])[1] for r in seq])
        lab = np.array([r["label"] for r in seq])
        st = np.array([r["state"] for r in seq])
        t = np.array([r["t"] for r in seq])
        clean = np.isin(phase, CLEAN)
        sw = np.mean([_switches_per_min(list(lab[(phase == p)]), t[phase == p]) for p in CLEAN])
        out[name] = {"acc_clean": float((lab[clean] == truth[clean]).mean()),
                     "flicker_per_min": float(sw),
                     "latency_s": float(np.nanmean([_latency(seq, b, l) for b, l in BOUNDARIES])),
                     "conflict_rate_masking": float((st[phase == "masking"] == "conflict").mean()),
                     "conflict_false_alarm_clean": float((st[clean] == "conflict").mean()),
                     "blend_in_joy_love": float(np.isin(lab[phase == "blend"], ["joy", "love"]).mean())}
    win = [r for r in recs if r["type"] == "window" and r["fused_mood"]]
    hits, n = 0, 0
    naive_hits = 0
    for r in win:
        lo = r["t"] - r["window_s"]
        ts = np.arange(max(0, lo), r["t"], 0.5)
        ph = [S.truth_at(x)[0] for x in ts]
        if not all(p in CLEAN for p in ph):
            continue
        tr = [S.truth_at(x)[1] for x in ts]
        maj = max(set(tr), key=tr.count)
        if tr.count(maj) / len(tr) < 0.6:          # near-tie windows have no well-defined single mood -> scored below
            continue
        hits += r["fused_mood"]["label"] == maj
        raw = [b["raw_label"] for b in recs if b["type"] == "block" and lo <= b["t"] < r["t"]]
        naive_hits += max(set(raw), key=raw.count) == maj if raw else 0
        n += 1
    # windows straddling a clean boundary (>= 30 % on each side): did the mood report the transition?
    tr_hits = []
    for r in win:
        lo = r["t"] - r["window_s"]
        for b, new in BOUNDARIES:
            if lo + 0.3 * r["window_s"] <= b <= r["t"] - 0.3 * r["window_s"]:
                fm = r["fused_mood"]
                tr_hits.append(fm["state"] == "transition" and fm.get("transition", [None, None])[1] == new)
    esc_hits = [("escalating" in r["behaviour"]["fused"]["tags"]) for r in win if 95 <= r["t"] <= 110]
    esc_fa = [("escalating" in r["behaviour"]["fused"]["tags"]) for r in win if r["t"] <= 45]
    out["window"] = {"n_clean_windows": n, "mood_acc": hits / n if n else float("nan"),
                     "naive_majority_vote_acc": naive_hits / n if n else float("nan"),
                     "transition_detected": float(np.mean(tr_hits)) if tr_hits else float("nan"),
                     "escalation_tag_rate": float(np.mean(esc_hits)) if esc_hits else float("nan"),
                     "escalation_false_alarm": float(np.mean(esc_fa)) if esc_fa else float("nan")}
    return out


def evaluate(seeds=range(10)) -> dict:
    runs = [evaluate_seed(s) for s in seeds]
    agg = {}
    for grp in runs[0]:
        agg[grp] = {k: {"mean": float(np.nanmean([r[grp][k] for r in runs])), "sd": float(np.nanstd([r[grp][k] for r in runs]))}
                    for k in runs[0][grp]}
    return {"synthetic": True, "n_seeds": len(runs), "summary": agg}


if __name__ == "__main__":
    import sys
    res = evaluate()
    json.dump(res, open(sys.argv[1] if len(sys.argv) > 1 else "results/temporal_eval.json", "w"), indent=2)
    for g, d in res["summary"].items():
        print(g, {k: f"{v['mean']:.3f}±{v['sd']:.3f}" for k, v in d.items()})
