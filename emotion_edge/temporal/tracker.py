"""Temporal models: from per-block predictions to (1) an instant emotion and (2) a mood over a window.

Two time scales, two probability models (rationale and references in docs/TEMPORAL_DESIGN.md):

1. INSTANT EMOTION  -- sticky hidden-Markov filter over the 8 canonical emotions.
   Emotions persist for seconds, so the state at t is strongly predicted by the state at t-dt.
     predict : b <- s(dt) * b + (1 - s(dt)) * pi        s(dt) = exp(-dt / dwell)   (time-aware stickiness)
     update  : b <- b * (q / pi) ** r, normalised       q = calibrated canonical posterior of the block,
                                                        r = block reliability (quality); q/pi = scaled likelihood
   Between observations only `predict` runs, so a stale modality decays smoothly back to the base rate pi
   and stops influencing fusion -- staleness handling falls out of the model instead of being a special case.

2. MOOD OVER A WINDOW -- Dirichlet evidence accumulation.
   The window's emotion *mixture* pi_W ~ Dirichlet(alpha),  alpha = alpha0 + sum_i w_i q_i
     w_i = r_i * min(1, dt_i / tau_c) * 0.5 ** (age_i / half_life)
   tau_c (correlation time) makes the evidence count roughly *independent* observations: 40 face frames in 10 s
   of an unchanged expression are not 40 votes. From the Dirichlet we get, in closed form or by sampling,
     mean mixture E[pi_W], 90% credible interval per emotion (Beta marginals), and
     P(emotion k is the dominant one) -- the probability that the reported mood is right *given the evidence*.

3. BEHAVIOURAL PATTERN -- dynamics of the filtered valence/arousal trajectory on a regular grid
   (mean, variability, instability = MSSD, inertia = lag-1 autocorrelation, trend, switch rate, dominant share),
   mapped to descriptive tags (stable / shifting / volatile / escalating / de-escalating / flat / incongruent).
   "escalating" = arousal rising while valence is not rising (towards anger/fear), "de-escalating" = arousal falling.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.stats import beta as beta_dist

from emotion_edge.labels import CANON, va_matrix
from emotion_edge.fusion.fuse import _VA_DIST, entropy_norm

K = len(CANON)
VA = va_matrix()


# ------------------------------------------------------------------------------------------------ instant
@dataclass
class StickyFilter:
    dwell_s: float = 4.0                         # expected persistence of an instant emotion
    base: np.ndarray = field(default_factory=lambda: np.full(K, 1.0 / K))
    b: np.ndarray = None
    t: float | None = None

    def __post_init__(self):
        self.b = self.base.copy()

    def predict(self, t: float):
        if self.t is not None and t > self.t:
            s = np.exp(-(t - self.t) / self.dwell_s)
            self.b = s * self.b + (1 - s) * self.base
        self.t = t if self.t is None else max(self.t, t)
        return self.b

    def update(self, t: float, q: np.ndarray, r: float = 1.0):
        self.predict(t)
        like = np.clip(q / self.base, 1e-9, None) ** float(np.clip(r, 0, 1))
        self.b = self.b * like
        self.b /= self.b.sum()
        return self.b

    @property
    def informativeness(self) -> float:
        """0 when the belief is the base rate (nothing known / stale), 1 when it is a point mass."""
        return float(1 - entropy_norm(self.b))


# ------------------------------------------------------------------------------------------------ mood
@dataclass
class Obs:
    t: float
    q: np.ndarray          # calibrated canonical distribution of one block
    r: float               # reliability of the block
    dur: float             # duration the block covers (s)


@dataclass
class MoodEstimate:
    alpha: np.ndarray
    n_eff: float
    mean: np.ndarray
    ci90: np.ndarray       # (K, 2) 90% credible interval of each emotion's share (Beta marginals)
    p_dominant: np.ndarray # P(class k has the largest share | evidence), Monte-Carlo over the Dirichlet
    label: str
    state: str             # dominant | leaning | mixed | conflicted | transition | insufficient
    second: str
    transition: tuple | None = None   # (from, to) when the window spans a change
    extra: dict = field(default_factory=dict)

    def as_dict(self):
        d = {"label": self.label, "state": self.state, "second": self.second, "n_eff": round(self.n_eff, 2),
             "p_dominant": round(float(self.p_dominant.max()), 3),
             "mixture": {c: round(float(m), 3) for c, m in zip(CANON, self.mean)},
             "ci90": {c: [round(float(a), 3), round(float(b), 3)] for c, (a, b) in zip(CANON, self.ci90)}}
        if self.transition:
            d["transition"] = list(self.transition)
        d.update(self.extra)
        return d


def summarise(alpha: np.ndarray, n_eff: float, alpha0: float, min_evidence=2.0, n_samples=2000, rng=None,
              halves: tuple | None = None) -> MoodEstimate:
    """Dirichlet(alpha) -> mean mixture, credible intervals, P(dominant), and a mood state.

    state rules (in order):
      insufficient : n_eff < min_evidence
      transition   : older and newer half of the window have different leading emotions, each with real evidence
      dominant     : P(top emotion is the largest share) >= 0.8
      mixed        : runner-up share >= 0.6 x top share and the two are valence/arousal-compatible
      conflicted   : same, but incompatible
      leaning      : otherwise (a leader exists but the evidence is not decisive yet)
    """
    rng = rng or np.random.default_rng(0)
    a0 = alpha.sum()
    mean = alpha / a0
    ci = np.stack([beta_dist.ppf([0.05, 0.95], a, a0 - a) for a in alpha])
    p_dom = np.bincount(rng.dirichlet(alpha, n_samples).argmax(1), minlength=K) / n_samples
    order = np.argsort(-mean)
    i1, i2 = int(order[0]), int(order[1])
    trans = None
    if n_eff < min_evidence:
        state = "insufficient"
    else:
        state = None
        if halves is not None:
            (ea, na), (eb, nb) = halves
            if na >= min_evidence / 2 and nb >= min_evidence / 2 and ea.argmax() != eb.argmax() \
                    and ea.max() > 1.5 * np.sort(ea)[-2] and eb.max() > 1.5 * np.sort(eb)[-2]:
                state, trans = "transition", (CANON[int(ea.argmax())], CANON[int(eb.argmax())])
        if state is None:
            if p_dom[i1] >= 0.8:
                state = "dominant"
            elif mean[i2] >= 0.6 * mean[i1]:
                state = "mixed" if _VA_DIST[i1, i2] < 0.75 else "conflicted"
            else:
                state = "leaning"
    return MoodEstimate(alpha, n_eff, mean, ci, p_dom, CANON[i1], state, CANON[i2], trans)


@dataclass
class MoodWindow:
    window_s: float = 30.0
    tau_c: float = 1.0                     # correlation time of the modality's observations
    half_life_s: float | None = None       # recency weighting inside the window (None = window_s)
    alpha0: float = 0.25                   # per-class prior pseudo-count (weak, symmetric)
    min_evidence: float = 1.5              # below this n_eff the mood is "insufficient"
    n_samples: int = 2000
    obs: list = field(default_factory=list)

    def add(self, o: Obs):
        self.obs.append(o)

    def weight(self, o: Obs, now: float) -> float:
        """reliability x independence (dur/tau_c, capped at 1) x recency x informativeness of the block.
        Informativeness min(1, 2 x (1 - normalised entropy)) keeps flat "don't know" outputs from voting -- important for text,
        whose 6-class model is forced to spread mass when the true state (e.g. neutral) is outside its label set."""
        hl = self.half_life_s or self.window_s
        return o.r * min(1.0, o.dur / self.tau_c) * 0.5 ** ((now - o.t) / hl) * min(1.0, 2 * (1 - entropy_norm(o.q)))

    def evidence(self, now: float, lo: float | None = None, hi: float | None = None):
        e, n = np.zeros(K), 0.0
        for o in self.obs:
            if (lo is None or o.t >= lo) and (hi is None or o.t < hi):
                w = self.weight(o, now)
                e += w * o.q
                n += w
        return e, n

    def estimate(self, now: float, rng=None) -> MoodEstimate:
        self.obs = [o for o in self.obs if o.t >= now - self.window_s]
        e, n = self.evidence(now)
        mid = now - self.window_s / 2
        halves = (self.evidence(now, hi=mid), self.evidence(now, lo=mid))
        return summarise(self.alpha0 + e, n, self.alpha0 * K, self.min_evidence, self.n_samples, rng, halves)


def fuse_moods(windows: dict[str, "MoodWindow"], now: float, weights: dict | None = None, rng=None):
    """Cross-modal mood = Dirichlet whose evidence is the (weighted) SUM of each modality's evidence.
    If the modalities are independent noisy views of the same underlying mixture, adding pseudo-counts is the exact
    Bayesian update, and modalities with more (informative) evidence automatically dominate.
    Also returns the cross-modal disagreement of the window: max pairwise JSD of modality means, weighted by evidence."""
    from emotion_edge.fusion.fuse import jsd
    weights = weights or {}
    alpha0 = next(iter(windows.values())).alpha0 if windows else 0.25
    E, N = np.zeros(K), 0.0
    halves_a, halves_b = [np.zeros(K), 0.0], [np.zeros(K), 0.0]
    means, rel = {}, {}
    for m, w in windows.items():
        lam = weights.get(m, 1.0)
        e, n = w.evidence(now, lo=now - w.window_s)
        if n <= 0:
            continue
        E += lam * e; N += lam * n
        mid = now - w.window_s / 2
        for h, (lo, hi) in ((halves_a, (now - w.window_s, mid)), (halves_b, (mid, None))):
            he, hn = w.evidence(now, lo=lo, hi=hi)
            h[0] += lam * he; h[1] += lam * hn
        means[m], rel[m] = (alpha0 + e) / (alpha0 * K + n), n / (n + 3.0)
    if N <= 0:
        return None
    est = summarise(alpha0 + E, N, alpha0 * K, rng=rng, halves=(tuple(halves_a), tuple(halves_b)))
    names = list(means)
    dis = 0.0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            dis = max(dis, jsd(means[names[i]], means[names[j]]) * min(rel[names[i]], rel[names[j]]))
    est.extra = {"cross_modal_jsd": round(dis, 3), "modality_leaders": {m: CANON[int(v.argmax())] for m, v in means.items()}}
    return est


# ------------------------------------------------------------------------------------------------ behaviour
def behaviour(times: np.ndarray, beliefs: np.ndarray, min_points: int = 8, thresholds=None) -> dict:
    """Dynamics of a filtered belief trajectory sampled on a regular grid (times in s, beliefs (T, K)).

    Metrics follow the affect-dynamics literature (variability, instability/MSSD, inertia), computed on the
    expected valence/arousal of the belief. Tags are descriptive heuristics with explicit thresholds,
    not clinical categories."""
    th = {"volatile_switch_per_min": 4.0, "volatile_mssd": 0.08, "stable_share": 0.7, "trend_per_min": 0.3,
          "flat_info": 0.15, "min_minutes_for_trend": 0.25, **(thresholds or {})}
    if len(times) < min_points:
        return {"n": int(len(times)), "tags": ["insufficient"]}
    va = beliefs @ VA
    v, a = va[:, 0], va[:, 1]
    info = 1 - np.array([entropy_norm(b) for b in beliefs])
    top = beliefs.argmax(1)
    confident_top = np.where(beliefs.max(1) > 0.5, top, -1)      # hysteresis: ignore near-ties
    seq = confident_top[confident_top >= 0]
    switches = int((np.diff(seq) != 0).sum()) if len(seq) > 1 else 0
    minutes = max((times[-1] - times[0]) / 60.0, 1e-6)
    tm = (times - times[0]) / 60.0

    def slope(y):
        return float(np.polyfit(tm, y, 1)[0]) if np.ptp(tm) > 0 else 0.0

    def ac1(y):
        y = y - y.mean()
        d = (y ** 2).sum()
        return float((y[1:] * y[:-1]).sum() / d) if d > 1e-12 else 1.0

    dom = np.bincount(top, minlength=K).argmax()
    m = {"n": int(len(times)), "minutes": round(minutes, 2),
         "valence_mean": float(v.mean()), "arousal_mean": float(a.mean()),
         "valence_sd": float(v.std()), "arousal_sd": float(a.std()),
         "valence_mssd": float(np.mean(np.diff(v) ** 2)), "arousal_mssd": float(np.mean(np.diff(a) ** 2)),
         "valence_inertia": ac1(v), "arousal_inertia": ac1(a),
         "valence_trend_per_min": slope(v), "arousal_trend_per_min": slope(a),
         "switch_per_min": float(switches / minutes), "dominant": CANON[int(dom)],
         "dominant_share": float((top == dom).mean()), "mean_informativeness": float(info.mean())}
    tags = []
    if m["mean_informativeness"] < th["flat_info"]:
        tags.append("flat")
    if m["switch_per_min"] > th["volatile_switch_per_min"] or max(m["valence_mssd"], m["arousal_mssd"]) > th["volatile_mssd"]:
        tags.append("volatile")
    trends_ok = m["minutes"] >= th["min_minutes_for_trend"]          # avoid start-up transients
    if trends_ok and m["arousal_trend_per_min"] > th["trend_per_min"] and m["valence_trend_per_min"] < th["trend_per_min"]:
        tags.append("escalating")
    elif trends_ok and m["arousal_trend_per_min"] < -th["trend_per_min"]:
        tags.append("de-escalating")
    if not tags or tags == ["flat"]:
        tags.append("stable" if m["dominant_share"] >= th["stable_share"] else "shifting")
    m["tags"] = tags
    return m
