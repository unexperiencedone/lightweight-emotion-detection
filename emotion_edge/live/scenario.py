"""Scripted synthetic session for demonstrating and evaluating the temporal layer (NOT real-world accuracy).

Model outputs are simulated per block as in fusion/simulate.py (strength x one-hot of what the modality 'sees' + noise).
The script covers the cases the temporal layer must handle:

  phase          t (s)     truth      what each modality sees
  neutral        0-20      neutral    all neutral (text has no neutral class -> uninformative)
  joy            20-45     joy        all joy
  masking        45-62     anger      text joy ("great, just great"), voice + face anger  -> incongruence
  sadness        62-85     sadness    all sadness; face missing 70-76 s (looked away)
  escalation     85-105    fear       sadness -> fear with rising intensity (arousal climbs)
  blend          105-130   joy/love   each block randomly joy or love
"""
from __future__ import annotations
import numpy as np

from emotion_edge.fusion.simulate import _logits
from emotion_edge.live.sources import Event

PHASES = [
    ("neutral", 0, 20, "neutral", {}),
    ("joy", 20, 45, "joy", {}),
    ("masking", 45, 62, "anger", {"text": "joy"}),
    ("sadness", 62, 85, "sadness", {}),
    ("escalation", 85, 105, "fear", {}),
    ("blend", 105, 130, "joy", {}),
]
DURATION = 130.0
FACE_GAP = (70.0, 76.0)


def truth_at(t):
    for name, a, b, truth, _ in PHASES:
        if a <= t < b:
            return name, truth
    return PHASES[-1][0], PHASES[-1][3]


def _seen(t, modality, rng):
    name, truth = truth_at(t)
    if name == "blend":
        return rng.choice(["joy", "love"])
    if name == "escalation":
        frac = (t - 85) / 20
        return "fear" if rng.random() < 0.3 + 0.7 * frac else "sadness"
    return dict(PHASES[[p[0] for p in PHASES].index(name)][4]).get(modality, truth)


def _strength(t):
    name, _ = truth_at(t)
    return 0.6 + 0.6 * (t - 85) / 20 if name == "escalation" else 1.0


def events(seed=0, face_fps=4.0):
    rng = np.random.default_rng(seed)
    ev = []
    # face: every frame at face_fps except during the look-away gap
    for t in np.arange(0, DURATION, 1 / face_fps):
        if FACE_GAP[0] <= t < FACE_GAP[1]:
            continue
        q = float(np.clip(rng.beta(6, 2), 0.2, 1))
        ev.append(Event(float(t), "block", {"modality": "face", "logits": _logits("face", _seen(t, "face", rng), rng, (0.6 + 0.4 * q) * _strength(t)),
                                             "quality": q, "dur": 1 / face_fps}))
    # speech: alternating voiced segments and pauses; a transcript utterance usually follows each segment
    t = 1.0
    while t < DURATION:
        dur = float(rng.uniform(1.2, 5.0))
        end = min(t + dur, DURATION)
        mid = (t + end) / 2
        q = float(np.clip(rng.beta(5, 2), 0.2, 1))
        ev.append(Event(end, "block", {"modality": "speech", "logits": _logits("speech", _seen(mid, "speech", rng), rng, (0.6 + 0.4 * q) * _strength(mid)),
                                        "quality": q, "dur": end - t}))
        if rng.random() < 0.85:
            n_words = int(rng.integers(2, 16))
            ev.append(Event(end + 0.3, "block", {"modality": "text", "logits": _logits("text", _seen(mid, "text", rng), rng, min(1.0, n_words / 6) * _strength(mid)),
                                                  "quality": min(1.0, n_words / 6), "dur": 1.0, "meta": {"n_words": n_words}}))
        t = end + float(rng.uniform(0.5, 3.0))
    return sorted(ev)
