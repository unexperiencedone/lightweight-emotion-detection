from __future__ import annotations
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax


def classification_report(y, pred, labels):
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=range(len(labels)), zero_division=0)
    return {"accuracy": float(accuracy_score(y, pred)),
            "accuracy_rounded_pct": round(100 * float(accuracy_score(y, pred)), 1),
            "macro_f1": float(f1_score(y, pred, average="macro")),
            "per_class": {l: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                          for i, l in enumerate(labels)},
            "confusion": confusion_matrix(y, pred, labels=range(len(labels))).tolist()}


def ece(probs, y, bins=15):
    conf, pred = probs.max(1), probs.argmax(1)
    edges, tot = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            tot += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(tot)


def fit_temperature(logits, y):
    """Single-parameter temperature scaling (Guo et al. 2017) on held-out logits."""
    nll = lambda T: -log_softmax(logits / T, axis=1)[np.arange(len(y)), y].mean()
    T = float(minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x)
    return T, {"nll_before": float(nll(1.0)), "nll_after": float(nll(T)),
               "ece_before": ece(softmax(logits, 1), y), "ece_after": ece(softmax(logits / T, 1), y)}
