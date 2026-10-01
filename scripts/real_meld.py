"""Real-data study of the TEXT temporal layer on MELD (Poria et al., 2019): multi-party TV conversations,
every utterance labelled by humans (7 emotions), with speaker and start/end timestamps.

Questions answered on REAL labelled sequences:
  1. How good is a lightweight text model on conversational text? (TF-IDF + logistic regression; pretrained transformers
     are unreachable from this sandbox, see docs)
  2. Does the sticky filter help *per utterance* in real conversations, and what dwell is best?  (tuned on dev, reported on test)
  3. Does the Dirichlet window mood recover a speaker's dominant emotion in a dialogue, and is P(dominant) calibrated?

    python scripts/real_meld.py --root /path/to/meld --out results/real_meld.json
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emotion_edge.labels import TEXT_CONV_LABELS, CANON
from emotion_edge.fusion.fuse import to_canonical
from emotion_edge.temporal.tracker import StickyFilter, MoodWindow, Obs
from emotion_edge.common.metrics import fit_temperature, ece
from scipy.special import softmax


def ts(s):
    h, m, rest = s.split(":")
    sec, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(sec) + int(ms) / 1000


def load(root):
    out = {}
    for split in ("train", "dev", "test"):
        d = pd.read_csv(Path(root) / f"{split}_sent_emo.csv", encoding="utf-8")
        d["Utterance"] = d.Utterance.str.replace("\x92", "'").str.replace("’", "'").str.replace("\x85", "...")
        d["y"] = d.Emotion.map(TEXT_CONV_LABELS.index)
        d["t_end"] = d.EndTime.map(ts)
        d["t_start"] = d.StartTime.map(ts)
        out[split] = d
    return out


def utterance_times(g):
    """Utterance end times relative to dialogue start. MELD timestamps are episode times and are occasionally
    non-monotonic; fall back to 3 s spacing where they go backwards or jump > 60 s."""
    t = g.t_end.to_numpy(float) - g.t_start.iloc[0]
    out, last = [], 0.0
    for i, x in enumerate(t):
        x = last + 3.0 if (i and (x <= last or x - last > 60)) or (i == 0 and (x < 0 or x > 60)) else x
        out.append(x); last = x
    return np.array(out)


def weighted_f1(y, p):
    from sklearn.metrics import f1_score
    return float(f1_score(y, p, average="weighted"))


def run_filter(data, logp_canon, dwell, by="speaker", carry=1.0):
    """Filtered prediction per utterance (state right after observing it). by='speaker' tracks each speaker separately
    (the person-centred view used in the live pipeline); by='dialogue' shares one state across speakers."""
    pred = np.empty(len(data), int)
    for _, g in data.groupby("Dialogue_ID", sort=False):
        times = utterance_times(g)
        filters = {}
        for (i, row), t in zip(g.iterrows(), times):
            key = row.Speaker if by == "speaker" else 0
            f = filters.setdefault(key, StickyFilter(dwell_s=dwell, carry=carry))
            f.update(t, logp_canon[data.index.get_loc(i)], 1.0)
            pred[data.index.get_loc(i)] = int(f.b.argmax())
    return pred


def canon_to_meld(idx):
    m = {CANON.index(c): TEXT_CONV_LABELS.index(c) for c in TEXT_CONV_LABELS}
    m[CANON.index("love")] = TEXT_CONV_LABELS.index("joy")
    return np.array([m[i] for i in idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="results/real_meld.json")
    a = ap.parse_args()
    D = load(a.root)
    tr, dv, te = D["train"], D["dev"], D["test"]
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline, make_union
    res = {"dataset": "MELD (text only)", "n": {k: len(v) for k, v in D.items()},
           "label_dist_test": te.Emotion.value_counts(normalize=True).round(3).to_dict()}

    # 1. text model: tune C on dev weighted-F1
    best = None
    for C in (0.5, 1, 2, 4, 8):
        for cw in (None, "balanced"):
            m = make_pipeline(make_union(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                                         TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True)),
                              LogisticRegression(C=C, max_iter=3000, class_weight=cw))
            m.fit(tr.Utterance, tr.y)
            f = weighted_f1(dv.y, m.predict(dv.Utterance))
            if best is None or f > best[0]:
                best = (f, C, cw, m)
    f_dev, C, cw, model = best
    res["text_model"] = {"type": "tfidf(word 1-2 + char 2-5) + logistic regression", "C": C, "class_weight": cw,
                         "dev_weighted_f1": f_dev}
    lv = np.log(model.predict_proba(dv.Utterance) + 1e-9)
    T, cal = fit_temperature(lv, dv.y.to_numpy())
    res["calibration"] = {"temperature": T, **cal}

    def canon_probs(d):
        p = softmax(np.log(model.predict_proba(d.Utterance) + 1e-9) / T, axis=1)
        return np.stack([to_canonical(x, "text_conv") for x in p]), p

    qd, pd_ = canon_probs(dv)
    qt, pt = canon_probs(te)
    raw_t = pt.argmax(1)
    res["test_raw"] = {"accuracy": float((raw_t == te.y).mean()), "weighted_f1": weighted_f1(te.y, raw_t),
                       "majority_class_baseline_acc": float((te.y == tr.y.mode()[0]).mean())}

    # 2. sticky filter on real conversations: tune dwell on dev
    grid = [0.5, 1, 2, 4, 8, 16, 32]
    sweep = {}
    for by in ("speaker", "dialogue"):
        for d in grid:
            for c in (0.0, 0.25, 0.5, 0.75, 1.0):
                p = canon_to_meld(run_filter(dv, qd, d, by, c))
                sweep[f"{by}_{d}_{c}"] = weighted_f1(dv.y, p)
    res["dev_sweep_weighted_f1"] = {"raw": weighted_f1(dv.y, pd_.argmax(1)), **sweep}
    res["dev_best_by_carry"] = {c: max(v for k, v in sweep.items() if k.endswith(f"_{c}")) for c in (0.0, 0.25, 0.5, 0.75, 1.0)}
    res["dev_best_by_dwell_at_carry1"] = {d: max(v for k, v in sweep.items() if k.endswith(f"_{d}_1.0")) for d in grid}
    best_key = max(sweep, key=sweep.get)
    by, d, c = best_key.split("_")[0], float(best_key.split("_")[1]), float(best_key.split("_")[2])
    p_f = canon_to_meld(run_filter(te, qt, d, by, c))
    res["test_filtered"] = {"chosen_on_dev": {"track": by, "dwell_s": d, "carry": c},
                            "accuracy": float((p_f == te.y).mean()), "weighted_f1": weighted_f1(te.y, p_f),
                            "switch_rate_raw": float(np.mean([np.mean(np.diff(raw_t[g.index - te.index[0]]) != 0) for _, g in te.groupby("Dialogue_ID") if len(g) > 1])),
                            }
    # human emotional inertia in MELD: how often does a speaker keep the same label on their next utterance?
    keep, n = 0, 0
    for _, g in pd.concat([tr, dv]).groupby("Dialogue_ID"):
        for _, s in g.groupby("Speaker"):
            y = s.y.to_numpy()
            keep += int((y[1:] == y[:-1]).sum()); n += max(0, len(y) - 1)
    res["human_same_label_next_utterance_same_speaker"] = keep / n

    # 3. window mood per (dialogue, speaker) with >= 4 utterances: Dirichlet over the whole dialogue
    rows = []
    for (dlg, spk), g in te.groupby(["Dialogue_ID", "Speaker"]):
        if len(g) < 4:
            continue
        y = g.y.to_numpy()
        counts = np.bincount(y, minlength=7)
        if np.sort(counts)[-1] == np.sort(counts)[-2]:
            continue                                           # no single dominant human label
        dlg_g = te[te.Dialogue_ID == dlg]
        times = dict(zip(dlg_g.index, utterance_times(dlg_g)))
        w = MoodWindow(window_s=1e6, tau_c=1.0, half_life_s=1e9)
        for i in g.index:
            w.add(Obs(times[i], qt[te.index.get_loc(i)], 1.0, 1.0))
        e = w.estimate(max(times.values()) + 1)
        lab = canon_to_meld([CANON.index(e.label)])[0]
        raw = raw_t[[te.index.get_loc(i) for i in g.index]]
        vote = np.bincount(raw, minlength=7).argmax()
        rows.append({"truth": int(counts.argmax()), "mood": int(lab), "vote": int(vote), "p_dom": float(e.p_dominant.max()),
                     "state": e.state, "n": len(g)})
    r = pd.DataFrame(rows)
    ok = (r.mood == r.truth).to_numpy()
    bins = np.linspace(0, 1, 6)
    rel = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (r.p_dom >= lo) & (r.p_dom < hi + (hi == 1))
        if m.any():
            rel.append({"bin": [round(lo, 1), round(hi, 1)], "n": int(m.sum()), "mean_p": float(r.p_dom[m].mean()), "acc": float(ok[m].mean())})
    res["mood"] = {"n_speaker_dialogues": len(r), "dirichlet_acc": float(ok.mean()), "raw_majority_vote_acc": float((r.vote == r.truth).mean()),
                   "majority_class_baseline": float((r.truth == 0).mean()),
                   "p_dominant_ece": float(np.sum([b["n"] * abs(b["acc"] - b["mean_p"]) for b in rel]) / len(r)),
                   "reliability": rel,
                   "acc_by_state": {s: {"n": int((r.state == s).sum()), "acc": float(ok[r.state == s].mean())} for s in r.state.unique()}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: v for k, v in res.items() if k not in ("dev_sweep_weighted_f1", "mood")}, indent=1, default=float))
    print("dev best by carry:", res["dev_best_by_carry"], "| by dwell at carry 1:", res["dev_best_by_dwell_at_carry1"])


if __name__ == "__main__":
    main()
