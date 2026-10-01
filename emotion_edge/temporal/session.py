"""LiveSession: per-modality temporal state + fusion on a shared clock.

    observe(modality, t, logits|probs, quality, dur)   -- one analysed block (utterance, voiced segment, face frame)
    tick(now)        -> instant record   (every `tick_s`, default 0.5 s)
    report(now)      -> window record    (every `hop_s`, default 5 s, over the last `window_s`, default 30 s)

Instant record: per-modality filtered belief + cross-modal fused instant emotion with ambiguity state.
Window record : per-modality mood (Dirichlet), fused mood, behavioural pattern of the fused and per-modality
                trajectories, and the share of the window spent in cross-modal conflict.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.special import softmax

from emotion_edge.labels import CANON
from emotion_edge.fusion.fuse import to_canonical, fuse_canonical, Thresholds
from emotion_edge.temporal.tracker import StickyFilter, MoodWindow, Obs, behaviour, fuse_moods

# Per-modality temporal parameters (see docs/TEMPORAL_DESIGN.md, "Parameter choices").
#   dwell_s : how long an instant reading stays relevant without new evidence
#   tau_c   : correlation time -> one "independent" unit of mood evidence per tau_c seconds of observation
DEFAULTS = {
    # text: persistence 10 s (a reading stays comparable with face/voice until the next utterance) but NO inertia:
    # on MELD conversations any carry > 0 lowers per-utterance weighted-F1 (docs/REAL_DATA_STUDY.md)
    "text":   {"dwell_s": 10.0, "carry": 0.0, "tau_c": 1.0, "base_weight": 1.0},   # one utterance = one unit (dur := 1)
    "speech": {"dwell_s": 6.0,  "carry": 1.0, "tau_c": 2.0, "base_weight": 1.0},   # 1-6 s segments
    "face":   {"dwell_s": 3.0,  "carry": 1.0, "tau_c": 1.0, "base_weight": 1.0},   # 4 fps frames, highly autocorrelated
}


@dataclass
class SessionConfig:
    window_s: float = 30.0
    hop_s: float = 5.0
    tick_s: float = 0.5
    temperatures: dict = field(default_factory=lambda: {"text": 1.0, "speech": 1.0, "face": 1.0})
    modality: dict = field(default_factory=lambda: {k: dict(v) for k, v in DEFAULTS.items()})
    thresholds: Thresholds = field(default_factory=Thresholds)
    p_incongruent: float = 0.4        # window tagged 'incongruent' when P(modality leaders incompatible) >= this


class LiveSession:
    def __init__(self, cfg: SessionConfig | None = None):
        self.cfg = cfg or SessionConfig()
        self.filters = {m: StickyFilter(dwell_s=p["dwell_s"], carry=p.get("carry", 1.0)) for m, p in self.cfg.modality.items()}
        self.moods = {m: MoodWindow(window_s=self.cfg.window_s, tau_c=p["tau_c"]) for m, p in self.cfg.modality.items()}
        self.seen: set[str] = set()
        self.grid_t: list[float] = []
        self.grid_fused: list[np.ndarray] = []
        self.grid_mod: dict[str, list[np.ndarray]] = {m: [] for m in self.cfg.modality}
        self.grid_state: list[str] = []
        self.raw: list[dict] = []

    # ---------------------------------------------------------------- input
    def observe(self, modality: str, t: float, logits=None, probs=None, quality: float = 1.0, dur: float = 1.0, meta=None):
        p = np.asarray(probs, float) if probs is not None else softmax(np.asarray(logits, float) / self.cfg.temperatures.get(modality, 1.0))
        q = to_canonical(p, modality)
        r = float(np.clip(quality * self.cfg.modality[modality]["base_weight"], 0, 1))
        self.filters[modality].update(t, q, r)
        self.moods[modality].add(Obs(t, q, r, dur))
        self.seen.add(modality)
        rec = {"type": "block", "t": round(t, 3), "modality": modality, "raw_label": CANON[int(q.argmax())],
               "raw_p": round(float(q.max()), 3), "quality": round(r, 3), "dur": round(dur, 2)}
        if meta:
            rec.update(meta)
        self.raw.append(rec)
        return rec

    # ---------------------------------------------------------------- instant
    def tick(self, now: float) -> dict:
        if not self.seen:
            return {"type": "instant", "t": now, "label": None, "state": "no_input"}
        qs, per = {}, {}
        for m in self.seen:
            f = self.filters[m]
            b = f.predict(now)
            self.grid_mod[m].append(b.copy())
            q = b / f.base
            qs[m] = q / q.sum()
            per[m] = {"top": CANON[int(b.argmax())], "p_top": round(float(b.max()), 3), "reliability": round(f.informativeness, 3)}
        for m in self.cfg.modality:
            if m not in self.seen:
                self.grid_mod[m].append(np.full(len(CANON), 1 / len(CANON)))
        fz = fuse_canonical(qs, per, th=self.cfg.thresholds)
        self.grid_t.append(now)
        self.grid_fused.append(fz.posterior)
        self.grid_state.append(fz.state)
        d = fz.as_dict()
        return {"type": "instant", "t": round(now, 3), **d}

    # ---------------------------------------------------------------- window
    def report(self, now: float) -> dict:
        lo = now - self.cfg.window_s
        moods = {m: self.moods[m].estimate(now) for m in self.seen}
        fused_mood = fuse_moods({m: self.moods[m] for m in self.seen}, now,
                                {m: p["base_weight"] for m, p in self.cfg.modality.items()})
        t = np.array(self.grid_t)
        sel = t >= lo
        beh = {"fused": behaviour(t[sel], np.array(self.grid_fused)[sel]) if sel.any() else {"tags": ["insufficient"]}}
        n_grid = len(self.grid_t)
        for m in self.seen:
            g = np.array(self.grid_mod[m][-n_grid:])
            beh[m] = behaviour(t[sel], g[sel]) if sel.any() else {"tags": ["insufficient"]}
        states = np.array(self.grid_state)[sel]
        conflict_share = float((states == "conflict").mean()) if len(states) else 0.0
        out = {"type": "window", "t": round(now, 3), "window_s": self.cfg.window_s,
               "moods": {m: e.as_dict() for m, e in moods.items()},
               "fused_mood": None if fused_mood is None else fused_mood.as_dict(),
               "behaviour": beh, "conflict_share": round(conflict_share, 3)}
        if conflict_share >= 0.3 or (fused_mood is not None and fused_mood.extra.get("p_incongruent", 0) >= self.cfg.p_incongruent):
            out["behaviour"]["fused"]["tags"] = out["behaviour"]["fused"]["tags"] + ["incongruent"]
        return out
