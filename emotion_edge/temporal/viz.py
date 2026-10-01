"""Figures for the temporal layer (timeline, mood, valence/arousal, filter effect, segmentation, Dirichlet, evaluation).

Colour: one fixed hue per canonical emotion across every figure (validated categorical palette; the low-contrast
slots are always paired with direct labels or a legend, never colour alone). Text stays in neutral ink.
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from emotion_edge.labels import CANON, va_matrix

EMO_COLOR = {"anger": "#e34948", "disgust": "#008300", "fear": "#4a3aa7", "joy": "#eda100", "love": "#e87ba4",
             "neutral": "#1baf7a", "sadness": "#2a78d6", "surprise": "#eb6834"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
STATE_COLOR = {"confident": "#52514e", "blend": "#e87ba4", "ambiguous": "#eda100", "conflict": "#e34948", "uncertain": "#b9b8b2"}
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "figure.dpi": 140,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.titlecolor": INK, "axes.titlesize": 10, "axes.titleweight": "bold", "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "savefig.facecolor": SURF, "grid.color": GRID, "legend.frameon": False})


def _runs(times, labels):
    """[(t0, t1, label)] collapsing consecutive equal labels."""
    out = []
    for t, l in zip(times, labels):
        if out and out[-1][2] == l:
            out[-1][1] = t
        else:
            out.append([t, t, l])
    return out


def _strip(ax, y, runs, h=0.8, step=0.5, label_min=6.0):
    for t0, t1, l in runs:
        if l is None:
            continue
        w = t1 - t0 + step
        ax.barh(y, w - 0.15, left=t0, height=h, color=EMO_COLOR[l], edgecolor=SURF, linewidth=1)   # gap = 2nd encoding
        if w >= label_min:
            ax.text(t0 + w / 2, y, l, ha="center", va="center", fontsize=7, color=INK)


def _emotion_legend(ax, loc="upper left", anchor=(1.0, 1.0)):
    ax.legend(handles=[Patch(color=EMO_COLOR[c], label=c) for c in CANON], loc=loc, bbox_to_anchor=anchor, fontsize=7)


def fig_timeline(recs, phases, path, title="Live session: instant emotion per modality and fused"):
    blocks = [r for r in recs if r["type"] == "block"]
    inst = [r for r in recs if r["type"] == "instant" and r.get("label")]
    fig, ax = plt.subplots(2, 1, figsize=(12, 5.6), sharex=True, gridspec_kw={"height_ratios": [3.2, 1.2]})
    rows = ["ground truth", "text  (raw blocks)", "voice (raw blocks)", "face  (raw frames)", "fused instant (filtered)"]
    a = ax[0]
    for name, t0, t1, truth, over in phases:
        lab = truth if name != "blend" else "joy"
        a.barh(4, t1 - t0 - 0.15, left=t0, height=0.8, color=EMO_COLOR[lab], edgecolor=SURF)
        txt = {"masking": "anger (masked)", "blend": "joy + love blend", "escalation": "sadness -> fear"}.get(name, truth)
        a.text((t0 + t1) / 2, 4, txt, ha="center", va="center", fontsize=7, color=INK)
    for y, m in ((3, "text"), (2, "speech"), (1, "face")):
        bb = [b for b in blocks if b["modality"] == m]
        a.scatter([b["t"] for b in bb], [y] * len(bb), c=[EMO_COLOR[b["raw_label"]] for b in bb], s=14 if m == "face" else 30,
                  marker="|" if m == "face" else "o", linewidths=1.5)
    _strip(a, 0, _runs([r["t"] for r in inst], [r["label"] for r in inst]))
    a.set_yticks(range(5)); a.set_yticklabels(rows[::-1]); a.set_ylim(-0.6, 4.6)
    a.set_title(title, loc="left")
    _emotion_legend(a)
    b = ax[1]
    t = np.array([r["t"] for r in inst]); p = np.array([r["p_top1"] for r in inst])
    b.plot(t, p, color=INK2, lw=1.2, label="fused p(top-1)")
    for s, c in STATE_COLOR.items():
        sel = np.array([r["state"] == s for r in inst])
        if sel.any():
            b.scatter(t[sel], np.full(sel.sum(), 1.07), color=c, s=10, marker="s", label=s)
    b.set_ylim(0, 1.15); b.set_ylabel("probability"); b.set_xlabel("time (s)")
    b.legend(loc="upper left", bbox_to_anchor=(1.0, 1.1), fontsize=7, title="ambiguity state", title_fontsize=7)
    b.grid(axis="y", lw=0.5)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_mood(recs, path):
    win = [r for r in recs if r["type"] == "window" and r["fused_mood"]]
    t = np.array([r["t"] for r in win])
    mix = np.array([[r["fused_mood"]["mixture"][c] for c in CANON] for r in win])
    fig, ax = plt.subplots(2, 1, figsize=(12, 5.2), sharex=True, gridspec_kw={"height_ratios": [2.4, 1]})
    ax[0].stackplot(t, mix.T, colors=[EMO_COLOR[c] for c in CANON], edgecolor=SURF, linewidth=0.8)
    for r in win[::2]:
        fm = r["fused_mood"]
        lbl = fm["label"] if fm["state"] != "transition" else f"{fm['transition'][0]}->{fm['transition'][1]}"
        ax[0].text(r["t"], 1.02, f"{lbl}\n{fm['state']}", ha="center", va="bottom", fontsize=6, color=INK2)
    ax[0].set_ylim(0, 1.18); ax[0].set_ylabel("mood mixture E[pi]")
    ax[0].set_title("Window mood (30 s window, every 5 s): Dirichlet mean mixture, label and state", loc="left")
    _emotion_legend(ax[0])
    pd_ = np.array([r["fused_mood"]["p_dominant"] for r in win])
    ax[1].plot(t, pd_, "o-", color=INK2, ms=3, lw=1.2)
    ax[1].axhline(0.8, ls="--", color=GRID, lw=1); ax[1].text(t[0], 0.82, "'dominant' threshold", fontsize=7, color=INK2)
    ax[1].set_ylim(0, 1.05); ax[1].set_ylabel("P(label is dominant)"); ax[1].set_xlabel("window end time (s)")
    ax[1].grid(axis="y", lw=0.5)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_va(recs, phases, path):
    inst = [r for r in recs if r["type"] == "instant" and r.get("posterior")]
    t = np.array([r["t"] for r in inst])
    P = np.array([[r["posterior"][c] for c in CANON] for r in inst])
    va = P @ va_matrix()
    fig, ax = plt.subplots(2, 1, figsize=(12, 4), sharex=True)
    for k, (name, lab) in enumerate((("valence", "expected valence"), ("arousal", "expected arousal"))):
        for nm, t0, t1, truth, _ in phases:
            ax[k].axvspan(t0, t1, color=EMO_COLOR["joy" if nm == "blend" else truth], alpha=0.10, lw=0)
            if k == 0:
                ax[k].text((t0 + t1) / 2, 0.95, nm, ha="center", fontsize=7, color=INK2)
        ax[k].plot(t, va[:, k], color=INK, lw=1.3)
        ax[k].axhline(0, color=GRID, lw=1); ax[k].set_ylim(-1, 1.1); ax[k].set_ylabel(lab)
    ax[0].set_title("Fused instant state on the valence/arousal plane (behaviour metrics are computed on these curves)", loc="left")
    ax[1].set_xlabel("time (s)")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_filter_effect(recs, path, modality="face", emotion="joy", span=(10, 55)):
    """Raw per-frame probability vs the sticky-filter belief for one emotion."""
    from emotion_edge.temporal.tracker import StickyFilter
    blocks = [r for r in recs if r["type"] == "block" and r["modality"] == modality and span[0] <= r["t"] <= span[1]]
    inst = [r for r in recs if r["type"] == "instant" and r.get("posterior") and span[0] <= r["t"] <= span[1]]
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.scatter([b["t"] for b in blocks], [b["raw_p"] if b["raw_label"] == emotion else 0 for b in blocks], s=9,
               color=EMO_COLOR[emotion], alpha=0.7, label=f"raw {modality} frame: p({emotion}) if top-1 else 0")
    ax.plot([r["t"] for r in inst], [r["posterior"][emotion] for r in inst], color=INK, lw=1.6, label=f"fused filtered belief p({emotion})")
    ax.axvline(20, color=INK2, ls="--", lw=1); ax.text(20.3, 1.0, "true change neutral -> joy", fontsize=7, color=INK2)
    ax.set_ylim(0, 1.08); ax.set_xlabel("time (s)"); ax.set_ylabel("probability")
    ax.set_title("Sticky HMM filter: noisy frame-level readings become a stable instant emotion", loc="left")
    ax.legend(loc="lower right", fontsize=7); ax.grid(axis="y", lw=0.5)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_segmentation(path):
    """Audio VAD and text block closure on a synthetic example (real segmenter code, synthetic input)."""
    from emotion_edge.temporal.segment import AudioSegmenter, TextSegmenter
    sr = 16000
    rng = np.random.default_rng(0)
    plan = [(0.0, 0.6, 0), (0.6, 2.4, 1), (2.4, 2.6, 0), (2.6, 4.0, 1), (4.0, 5.2, 0), (5.2, 6.0, 1), (6.0, 7.5, 0), (7.5, 15.0, 1), (15.0, 16.0, 0)]
    y = []
    for a, b, v in plan:
        n = int((b - a) * sr); tt = np.arange(n) / sr
        y.append(0.003 * rng.normal(size=n) + v * 0.3 * np.sin(2 * np.pi * 160 * tt) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * tt)) ** 2)
    y = np.concatenate(y).astype(np.float32)
    seg = AudioSegmenter()
    blocks = []
    for i in range(0, len(y), 1600):
        blocks += seg.push(y[i:i + 1600])
    blocks += seg.flush()
    fig, ax = plt.subplots(2, 1, figsize=(12, 4.6), gridspec_kw={"height_ratios": [1.6, 1]})
    t = np.arange(len(y)) / sr
    ax[0].plot(t[::20], y[::20], color=INK2, lw=0.4)
    for b in blocks:
        ax[0].axvspan(b.t_start, b.t_end, color=EMO_COLOR["sadness"], alpha=0.18, lw=0)
        ax[0].text((b.t_start + b.t_end) / 2, 0.36, f"{b.t_end - b.t_start:.1f}s\n{b.reason}", ha="center", fontsize=7, color=INK)
    ax[0].set_ylim(-0.4, 0.5); ax[0].set_ylabel("waveform"); ax[0].set_xlabel("time (s)")
    ax[0].set_title("Audio blocks: energy VAD closes on a >=0.4 s pause, keeps short gaps (prosody), force-splits at 6 s, drops <0.8 s", loc="left")
    words = [("i", 0.2), ("really", 0.5), ("thought", 0.8), ("this", 1.1), ("would", 1.3), ("work.", 1.6), ("ok", 4.0),
             ("but", 4.3), ("now", 4.6), ("everything", 4.9), ("is", 5.2), ("broken", 5.5), ("and", 7.4), ("nobody", 7.7),
             ("cares", 8.0), ("at", 8.3), ("all", 8.6)]
    ts = TextSegmenter()
    tb = []
    for w, tt in words:
        tb += ts.push_word(w, tt)
    tb += ts.tick(12.0)
    for w, tt in words:
        ax[1].text(tt, 0.25, w, rotation=35, fontsize=7, color=INK, ha="left", va="bottom")
    for i, b in enumerate(tb):
        ax[1].barh(0, b.t_end - b.t_start + 0.3, left=b.t_start, height=0.18, color=EMO_COLOR["fear"], alpha=0.35)
        ax[1].text(b.t_start, -0.18 - 0.14 * (i % 2), f"block {i + 1}: {b.reason}, q={b.quality:.2f}", fontsize=7, color=INK2)
    ax[1].set_xlim(ax[0].get_xlim()); ax[1].set_ylim(-0.45, 0.9); ax[1].set_yticks([]); ax[1].set_xlabel("time (s)")
    ax[1].set_title("Text blocks: sentence end (>=3 words), pause >=1.2 s, 40-word cap; 'ok' (1 word) is merged forward", loc="left")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return blocks, tb


def fig_dirichlet(rec, path):
    """Mood credible intervals for one window record: per modality and fused."""
    groups = {**{m: v for m, v in rec["moods"].items()}, "fused": rec["fused_mood"]}
    fig, axs = plt.subplots(1, len(groups), figsize=(3.0 * len(groups), 3.2), sharey=True)
    for ax, (g, d) in zip(np.atleast_1d(axs), groups.items()):
        y = np.arange(len(CANON))
        for i, c in enumerate(CANON):
            lo, hi = d["ci90"][c]
            ax.plot([lo, hi], [i, i], color=EMO_COLOR[c], lw=2.5, solid_capstyle="round")
            ax.plot(d["mixture"][c], i, "o", color=EMO_COLOR[c], ms=6, mec=SURF, mew=1.5)
        ax.set_yticks(y); ax.set_yticklabels(CANON); ax.set_xlim(0, 1); ax.grid(axis="x", lw=0.5)
        ax.set_title(f"{g}: {d['label']} ({d['state']})\nP(dominant)={d['p_dominant']:.2f}, n_eff={d['n_eff']:.1f}", fontsize=8, loc="left")
        ax.set_xlabel("share of window")
    fig.suptitle(f"Window ending t={rec['t']:.0f} s: mean share and 90% credible interval per emotion", x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_eval(res, path):
    s = res["summary"]
    items = [("acc_clean", "instant accuracy\n(clean phases)", (0, 1)), ("flicker_per_min", "label switches / min\n(lower is better)", None),
             ("latency_s", "transition latency (s)\n(lower is better)", None), ("conflict_rate_masking", "masking flagged\nas 'conflict'", (0, 1)),
             ("conflict_false_alarm_clean", "'conflict' false alarms\n(clean phases)", (0, 1))]
    fig, axs = plt.subplots(1, len(items), figsize=(13, 2.9))
    for ax, (k, title, lim) in zip(axs, items):
        vals = [s["baseline"][k]["mean"], s["temporal"][k]["mean"]]
        sds = [s["baseline"][k]["sd"], s["temporal"][k]["sd"]]
        ax.bar([0, 1], vals, yerr=sds, color=["#b9b8b2", EMO_COLOR["sadness"]], width=0.6, capsize=3, error_kw={"ecolor": INK2, "lw": 1})
        top = max(v + d for v, d in zip(vals, sds))
        for i, (v, d) in enumerate(zip(vals, sds)):
            ax.text(i + 0.33, v, f"{v:.2f}", ha="left", va="center", fontsize=8, color=INK)
        if not lim:
            ax.set_ylim(0, top * 1.15)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["static\nfusion", "temporal\nlayer"], fontsize=8); ax.set_xlim(-0.5, 1.9)
        ax.set_title(title, fontsize=8, loc="left")
        if lim:
            ax.set_ylim(*lim)
    fig.suptitle(f"Temporal layer vs static per-tick fusion (synthetic session, {res['n_seeds']} seeds, mean ± sd)", x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
