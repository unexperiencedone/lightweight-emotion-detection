"""Probabilistic late fusion with explicit ambiguity typing.

Why late + probabilistic (not early fusion): the three modalities arrive at different rates, can drop out
(no face in frame, silence, no transcript), use different label sets and are trained on different corpora.
Each unimodal model therefore stays independent, is *calibrated* (temperature scaling), projected to a
shared 8-class canonical space, discounted by how much we trust it for *this* input, and combined as a
weighted log-opinion pool (a product of experts). Because the pool keeps the full posterior we can also ask
"how much do the experts disagree?" -- the signal early fusion throws away.

Pipeline per modality m with calibrated distribution p_m over its own labels:
  1. project:   q_m = p_m @ MAP_m                          (modality labels -> canonical)
  2. coverage:  classes m cannot express get the uniform level 1/|covered| (= "no information", not "impossible")
  3. discount:  q~_m = r_m q_m + (1-r_m) / K               (r_m in [0,1]: input quality x model reliability)
  4. pool:      log P(y) = log prior(y) + sum_m log q~_m(y) - log Z       (influence of m is controlled by r_m via step 3)

Ambiguity types (what a downstream policy should do about each):
  confident  - one class dominates and experts agree                        -> act
  blend      - top-2 are close AND valence/arousal-compatible (joy+love)     -> report mixed emotion
  ambiguous  - top-2 are close but incompatible (joy vs anger)               -> ask / wait for more evidence
  conflict   - experts disagree (JSD high) even if the pooled posterior looks peaked
               -> incongruence: sarcasm, masking, acted smile, or a failing sensor
  uncertain  - posterior near-uniform                                         -> abstain
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.special import softmax, rel_entr

from emotion_edge.labels import CANON, MAPS, LABELS, coverage, va_matrix

K = len(CANON)
_VA = va_matrix()
_VA_DIST = np.linalg.norm(_VA[:, None] - _VA[None], axis=-1)


@dataclass
class Thresholds:
    conf_p: float = 0.55          # min pooled top-1 probability to call "confident"
    margin: float = 0.20          # top1 - top2 below this => close competition
    entropy: float = 0.80         # normalised entropy above this => uncertain
    conflict_jsd: float = 0.35    # max pairwise JSD (normalised to [0,1]) above this => conflict
    compat_dist: float = 0.75     # V/A distance below which two emotions are a plausible blend


@dataclass
class Modality:
    name: str                      # 'text' | 'speech' | 'face'
    logits: np.ndarray | None = None
    probs: np.ndarray | None = None
    temperature: float = 1.0       # fitted on validation logits (common.metrics.fit_temperature)
    quality: float = 1.0           # per-input quality in [0,1]: SNR/voiced ratio, face-detector score, token count
    base_weight: float = 1.0       # per-modality prior reliability, e.g. from val accuracy / calibration


def _norm_probs(m: Modality) -> np.ndarray:
    if m.probs is not None:
        return np.asarray(m.probs, float)
    return softmax(np.asarray(m.logits, float) / m.temperature)


def to_canonical(p: np.ndarray, modality: str) -> np.ndarray:
    q = p @ MAPS[modality]
    cov = coverage(modality)
    q = np.where(cov, q, 1.0 / cov.sum())      # uncovered classes: uniform level => uninformative
    return q / q.sum()


def discount(q: np.ndarray, r: float) -> np.ndarray:
    return r * q + (1 - r) / K


def entropy_norm(p):
    p = np.clip(p, 1e-12, 1)
    return float(-(p * np.log(p)).sum() / np.log(len(p)))


def jsd(p, q):
    m = 0.5 * (p + q)
    return float((0.5 * rel_entr(p, m).sum() + 0.5 * rel_entr(q, m).sum()) / np.log(2))  # in [0,1]


@dataclass
class Fused:
    posterior: np.ndarray
    label: str
    state: str
    top2: tuple
    p_top1: float
    margin: float
    entropy: float
    conflict: float
    per_modality: dict = field(default_factory=dict)
    explanation: str = ""

    def as_dict(self):
        return {"label": self.label, "state": self.state, "p_top1": round(self.p_top1, 3), "top2": self.top2,
                "margin": round(self.margin, 3), "entropy": round(self.entropy, 3), "conflict": round(self.conflict, 3),
                "per_modality": self.per_modality, "explanation": self.explanation,
                "posterior": {c: round(float(p), 4) for c, p in zip(CANON, self.posterior)}}


def fuse(mods: list[Modality], prior: np.ndarray | None = None, th: Thresholds = Thresholds()) -> Fused:
    mods = [m for m in mods if m is not None and (m.logits is not None or m.probs is not None)]
    if not mods:
        raise ValueError("no modality available")
    logp = np.log(prior if prior is not None else np.full(K, 1.0 / K))
    qs, per = {}, {}
    for m in mods:
        p = _norm_probs(m)
        q = to_canonical(p, m.name)
        r = float(np.clip(m.quality * m.base_weight, 0, 1))
        qd = discount(q, r)
        qs[m.name] = qd
        logp = logp + np.log(qd)               # discounting already scales the influence; weight folded into r
        per[m.name] = {"top": CANON[int(q.argmax())], "p_top": round(float(q.max()), 3), "reliability": round(r, 3),
                       "entropy": round(entropy_norm(p), 3)}
    post = softmax(logp)
    order = np.argsort(-post)
    i1, i2 = int(order[0]), int(order[1])
    p1, margin, H = float(post[i1]), float(post[i1] - post[i2]), entropy_norm(post)
    # inter-expert disagreement: worst pairwise JSD among *informative* experts, weighted by min reliability
    names = list(qs)
    conflict = 0.0
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            ra, rb = per[names[a]]["reliability"], per[names[b]]["reliability"]
            conflict = max(conflict, jsd(qs[names[a]], qs[names[b]]) * min(1.0, 2 * min(ra, rb)))
    compat = _VA_DIST[i1, i2] < th.compat_dist
    if conflict > th.conflict_jsd and len(mods) > 1:
        state = "conflict"
    elif H > th.entropy or p1 < 0.30:
        state = "uncertain"
    elif margin < th.margin:
        state = "blend" if compat else "ambiguous"
    elif p1 >= th.conf_p:
        state = "confident"
    else:
        state = "uncertain"
    expl = {"confident": f"{CANON[i1]} supported by {', '.join(names)}",
            "blend": f"mixed {CANON[i1]}/{CANON[i2]} (valence-arousal compatible)",
            "ambiguous": f"{CANON[i1]} vs {CANON[i2]} are incompatible; gather more evidence",
            "conflict": "modalities disagree: " + "; ".join(f"{n}->{per[n]['top']}" for n in names),
            "uncertain": "no emotion is supported strongly enough; abstain"}[state]
    return Fused(post, CANON[i1], state, (CANON[i1], CANON[i2]), p1, margin, H, conflict, per, expl)


def risk_coverage(correct: np.ndarray, score: np.ndarray, n_points=20):
    """Selective-prediction curve: accuracy among the most-confident `coverage` fraction.
    score = anything where higher = more confident (e.g. p_top1 - conflict)."""
    order = np.argsort(-score)
    cs = np.cumsum(correct[order]) / (np.arange(len(correct)) + 1)
    cov = (np.arange(len(correct)) + 1) / len(correct)
    idx = np.linspace(0, len(correct) - 1, n_points).astype(int)
    return cov[idx], cs[idx]
