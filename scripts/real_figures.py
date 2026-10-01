"""Figures for the real-data study (CREMA-D + MELD), built from results/real_cremad.json, results/real_meld.json and
the saved logits. Writes docs/figures/real_*.png.

    python scripts/real_figures.py --root /path/to/cremad --feats /path/to/cremad_feats --models /path/to/cremad_models
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from emotion_edge.temporal.viz import EMO_COLOR, INK, INK2, GRID, SURF, STATE_COLOR, _runs, _strip, _emotion_legend
from emotion_edge.realdata import cremad as C

FIG = Path("docs/figures")
GREY, BLUE, ORANGE = "#b9b8b2", "#2a78d6", "#eb6834"
NAMES6 = C.NAMES


def bars_vs_humans(r):
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    groups = [("voice only", r["speech"]["human_acc_vs_intended_same_clips"], r["speech"]["test_acc_vs_intended"]),
              ("face only", r["face"]["human_acc_vs_intended_same_clips"], r["face"]["test_acc_vs_intended"]),
              ("voice + face", r["fusion_clip_level"]["test"]["human_av_acc_vs_intended"], r["fusion_clip_level"]["test"]["fused_acc_vs_intended"])]
    x = np.arange(len(groups))
    for k, (lab, col) in enumerate((("human raters (majority vote)", GREY), ("this system (int8-ready models)", BLUE))):
        vals = [g[1 + k] for g in groups]
        ax.bar(x + (k - 0.5) * 0.36, vals, 0.34, color=col, label=lab)
        for xi, v in zip(x, vals):
            ax.text(xi + (k - 0.5) * 0.36, v + 0.01, f"{v:.2f}", ha="center", fontsize=8, color=INK)
    ax.axhline(1 / 6, ls="--", color=GRID); ax.text(2.45, 1 / 6 + 0.01, "chance", fontsize=7, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels([g[0] for g in groups]); ax.set_ylim(0, 1); ax.set_ylabel("accuracy vs acted emotion")
    ax.set_title("CREMA-D, 15 unseen test actors: models vs human perception of the same clips", loc="left")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "real_vs_humans.png"); plt.close(fig)


def confusions(r):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, title in zip(axs, ("speech", "face"), ("voice prosody model", "face model (clip = mean of frames)")):
        cm = np.array(r[key]["confusion"], float); cm /= cm.sum(1, keepdims=True)
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        for i in range(6):
            for j in range(6):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if cm[i, j] > .5 else INK)
        ax.set_xticks(range(6)); ax.set_xticklabels(NAMES6, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(6)); ax.set_yticklabels(NAMES6, fontsize=8)
        ax.set_xlabel("predicted"); ax.set_ylabel("acted"); ax.set_title(f"{title}: acc {r[key]['test_acc_vs_intended']:.2f}", loc="left", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "real_confusions.png"); plt.close(fig)


def ambiguity(r):
    f = r["fusion_clip_level"]["test"]; s = r["selective_test"]["real_tuned"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.4))
    names = ["1 - p(top) predicts\nhumans ambiguous", "entropy predicts\nhumans ambiguous", "conflict predicts\nhuman voice/face\ndisagreement", "1 - p(top) predicts\nmodel error"]
    vals = [f["auroc_1_minus_p_top1__human_ambiguous"], f["auroc_entropy__human_ambiguous"], f["auroc_conflict__human_incongruent"] or np.nan, f["auroc_1_minus_p_top1__fused_wrong"]]
    axs[0].bar(range(4), vals, color=BLUE, width=0.55)
    for i, v in enumerate(vals):
        axs[0].text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    axs[0].axhline(0.5, ls="--", color=GRID); axs[0].text(3.3, 0.51, "chance", fontsize=7, color=INK2)
    axs[0].set_xticks(range(4)); axs[0].set_xticklabels(names, fontsize=7); axs[0].set_ylim(0, 1); axs[0].set_ylabel("AUROC")
    axs[0].set_title("Do the fusion's uncertainty signals match humans? (test)", loc="left", fontsize=9)
    labels = ["answered\n(confident)", "flagged"]
    acc = [s["acc_confident"], s["acc_flagged"]]; amb = [s["human_ambiguous_rate_confident"], s["human_ambiguous_rate_flagged"]]
    x = np.arange(2)
    axs[1].bar(x - 0.18, acc, 0.34, color=BLUE, label="model accuracy")
    axs[1].bar(x + 0.18, amb, 0.34, color=ORANGE, label="share humans found ambiguous")
    for xi, a, b in zip(x, acc, amb):
        axs[1].text(xi - 0.18, a + 0.01, f"{a:.2f}", ha="center", fontsize=8); axs[1].text(xi + 0.18, b + 0.01, f"{b:.2f}", ha="center", fontsize=8)
    axs[1].set_xticks(x); axs[1].set_xticklabels([f"{l}\n{c:.0%} of clips" for l, c in zip(labels, [s["coverage_confident"], 1 - s["coverage_confident"]])], fontsize=8)
    axs[1].set_ylim(0, 1); axs[1].legend(fontsize=7, loc="upper right")
    axs[1].set_title("Thresholds tuned on real val actors, applied to test", loc="left", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "real_ambiguity.png"); plt.close(fig)


def temporal_sweep(r):
    sw = r["temporal"]["val_sweep"]
    fd = sorted({s["face_dwell"] for s in sw}); sd = sorted({s["speech_dwell"] for s in sw}); tcs = sorted({s["face_tau_c"] for s in sw})
    tc = r["temporal"]["chosen_on_val"]["face"]["tau_c"]
    M = np.array([[next(s["instant_acc"] for s in sw if s["face_dwell"] == a and s["speech_dwell"] == b and s["face_tau_c"] == tc) for b in sd] for a in fd])
    t = r["temporal"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1, 1.3]})
    im = axs[0].imshow(M, cmap="Blues", aspect="auto")
    for i in range(len(fd)):
        for j in range(len(sd)):
            axs[0].text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=8, color="white" if M[i, j] > M.mean() else INK)
    axs[0].set_xticks(range(len(sd))); axs[0].set_xticklabels([f"{x:g}" for x in sd]); axs[0].set_yticks(range(len(fd))); axs[0].set_yticklabels([f"{x:g}" for x in fd])
    axs[0].set_xlabel("voice dwell (s)"); axs[0].set_ylabel("face dwell (s)")
    axs[0].set_title("Instant accuracy on VAL actors' real sessions", loc="left", fontsize=9)
    rows = [("static fusion\n(no temporal layer)", t["test_static_baseline"]), ("temporal layer,\ndefault params", t["test_default_params"]), ("temporal layer,\nretuned on real val", t["test_tuned_params"])]
    x = np.arange(len(rows))
    axs[1].bar(x, [v["instant_acc"] for _, v in rows], 0.55, color=[GREY, "#8fb3e0", BLUE])
    for xi, (_, v) in zip(x, rows):
        axs[1].text(xi, v["instant_acc"] + 0.01, f"acc {v['instant_acc']:.3f}\n{v['flicker_per_min']:.1f} switches/min", ha="center", fontsize=7.5)
    axs[1].set_xticks(x); axs[1].set_xticklabels([r_[0] for r_ in rows], fontsize=8); axs[1].set_ylim(0, 1.05); axs[1].set_ylabel("instant accuracy")
    axs[1].set_title("TEST actors (reported once)", loc="left", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "real_temporal.png"); plt.close(fig)


def reliability(r, meld):
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.plot([0, 1], [0, 1], ls="--", color=GRID)
    for rel, lab, col in ((meld["mood"]["reliability"], f"MELD speaker mood (ECE {meld['mood']['p_dominant_ece']:.3f})", ORANGE),
                          (r["temporal"]["test_tuned_params"]["mood_reliability"], f"CREMA-D session mood (ECE {r['temporal']['test_tuned_params']['mood_p_dominant_ece']:.3f})", BLUE)):
        ax.plot([b["mean_p"] for b in rel], [b["acc"] for b in rel], "o-", color=col, label=lab, ms=5)
        for b in rel:
            ax.text(b["mean_p"] + 0.01, b["acc"] - 0.04, f"n={b['n']}", fontsize=6, color=INK2)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.set_xlabel("P(dominant) reported with the mood"); ax.set_ylabel("observed accuracy of the mood label")
    ax.set_title("Is the mood probability honest? (real labels)", loc="left", fontsize=9); ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig(FIG / "real_mood_reliability.png"); plt.close(fig)


def meld_heat(meld):
    s = meld["dev_sweep_weighted_f1"]
    dw = [0.5, 1, 2, 4, 8, 16, 32]; cs = [0.0, 0.25, 0.5, 0.75, 1.0]
    M = np.array([[s[f"speaker_{d}_{c}"] for d in dw] for c in cs])
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ax.imshow(M, cmap="Blues", aspect="auto")
    for i in range(len(cs)):
        for j in range(len(dw)):
            ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=7.5, color="white" if M[i, j] > 0.515 else INK)
    ax.set_xticks(range(len(dw))); ax.set_xticklabels([f"{d:g}" for d in dw]); ax.set_yticks(range(len(cs))); ax.set_yticklabels([f"{c:g}" for c in cs])
    ax.set_xlabel("text dwell / persistence (s)"); ax.set_ylabel("carry / inertia")
    ax.set_title(f"MELD dev weighted-F1 per utterance (raw, no filter: {s['raw']:.3f}). Inertia hurts; persistence alone is free.", loc="left", fontsize=8.5)
    fig.tight_layout(); fig.savefig(FIG / "real_meld_text_inertia.png"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root"); ap.add_argument("--feats"); ap.add_argument("--models")
    a = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    r = json.loads(Path("results/real_cremad.json").read_text())
    meld = json.loads(Path("results/real_meld.json").read_text())
    bars_vs_humans(r); confusions(r); ambiguity(r); temporal_sweep(r); reliability(r, meld); meld_heat(meld)
    if a.root and a.feats and a.models:
        timeline(a, r)
    print("figures written")


def timeline(a, r):
    """Re-run one TEST actor's real session with the parameters retuned on val, and plot it."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from real_cremad import build_sessions
    from emotion_edge.live.pipeline import LivePipeline
    from emotion_edge.temporal.session import SessionConfig
    from emotion_edge.temporal import viz
    df = C.metadata(a.root); df["split"] = C.actor_split(df)
    A = dict(np.load(Path(a.feats) / "audio.npz")); F = dict(np.load(Path(a.feats) / "faces.npz"))
    Ls = np.load(Path(a.models) / "speech_logits.npy"); Lc = np.load(Path(a.models) / "face_frame_logits.npy")
    s = build_sessions(df, "test", A, F, Ls, Lc, np.random.default_rng(1))[0]
    temps = {"speech": r["speech"]["temperature"], "face": r["face"]["frame_temperature"], "text": 1.0}
    cfg = SessionConfig(window_s=20.0, temperatures=temps, modality=r["temporal"]["chosen_on_val"])
    recs = LivePipeline(cfg).run(s["events"])
    end = 120.0
    recs = [x for x in recs if x["t"] <= end]
    phases = []
    for t0, t1, l in s["truth"]:
        if t0 > end:
            break
        if phases and phases[-1][3] == l and t0 - phases[-1][2] < 2:
            phases[-1] = (phases[-1][0], phases[-1][1], t1, l, {})
        else:
            phases.append((l, t0, t1, l, {}))
    viz.fig_timeline(recs, phases, FIG / "real_session_timeline.png",
                     title=f"REAL recordings: CREMA-D test actor {s['actor']} (clips arranged in emotion episodes), first {end:.0f} s")


if __name__ == "__main__":
    main()
