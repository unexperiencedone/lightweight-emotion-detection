"""Label spaces and the mapping between them.

Every modality has its *own* label set (the datasets disagree), and fusion happens in a
shared canonical space.  Mapping matrices are rows = modality label, cols = canonical label.
Classes a modality cannot express (e.g. text has no "neutral") are marked uncovered so the
fusion step treats them as "no information" rather than "impossible".
"""
from __future__ import annotations
import numpy as np

# dair-ai/emotion label ids 0..5 (fixed by the dataset card)
TEXT_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
# RAVDESS-style speech set (CREMA-D is a subset: no calm / surprised)
SPEECH_LABELS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
# FER2013 / FER+ order
FACE_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# MELD conversational text (Poria et al., 2019): utterances from TV dialogues
TEXT_CONV_LABELS = ["neutral", "joy", "sadness", "anger", "surprise", "fear", "disgust"]

CANON = ["anger", "disgust", "fear", "joy", "love", "neutral", "sadness", "surprise"]
CIDX = {c: i for i, c in enumerate(CANON)}

# (valence, arousal) in [-1, 1]; coarse circumplex placement used only to decide whether two
# competing emotions are a plausible *blend* (joy+love) or genuinely *incompatible* (joy vs anger).
VALENCE_AROUSAL = {
    "anger": (-0.6, 0.8), "disgust": (-0.6, 0.4), "fear": (-0.7, 0.7), "joy": (0.8, 0.5),
    "love": (0.8, 0.3), "neutral": (0.0, 0.0), "sadness": (-0.7, -0.4), "surprise": (0.2, 0.8),
}

_TEXT_MAP = {"sadness": {"sadness": 1}, "joy": {"joy": 1}, "love": {"love": 1},
             "anger": {"anger": 1}, "fear": {"fear": 1}, "surprise": {"surprise": 1}}
_SPEECH_MAP = {"neutral": {"neutral": 1}, "calm": {"neutral": 1}, "happy": {"joy": .85, "love": .15},
               "sad": {"sadness": 1}, "angry": {"anger": 1}, "fearful": {"fear": 1},
               "disgust": {"disgust": 1}, "surprised": {"surprise": 1}}
_FACE_MAP = {"angry": {"anger": 1}, "disgust": {"disgust": 1}, "fear": {"fear": 1},
             "happy": {"joy": .85, "love": .15}, "sad": {"sadness": 1},
             "surprise": {"surprise": 1}, "neutral": {"neutral": 1}}
_TEXT_CONV_MAP = {"neutral": {"neutral": 1}, "joy": {"joy": .85, "love": .15}, "sadness": {"sadness": 1},
                  "anger": {"anger": 1}, "surprise": {"surprise": 1}, "fear": {"fear": 1}, "disgust": {"disgust": 1}}


def _matrix(labels, mapping):
    m = np.zeros((len(labels), len(CANON)))
    for i, l in enumerate(labels):
        for c, w in mapping[l].items():
            m[i, CIDX[c]] = w
    return m


MAPS = {"text": _matrix(TEXT_LABELS, _TEXT_MAP),
        "speech": _matrix(SPEECH_LABELS, _SPEECH_MAP),
        "face": _matrix(FACE_LABELS, _FACE_MAP),
        "text_conv": _matrix(TEXT_CONV_LABELS, _TEXT_CONV_MAP)}
LABELS = {"text": TEXT_LABELS, "speech": SPEECH_LABELS, "face": FACE_LABELS, "text_conv": TEXT_CONV_LABELS}


def coverage(modality: str) -> np.ndarray:
    """Boolean mask over CANON of classes the modality can emit."""
    return MAPS[modality].sum(0) > 0


def va_matrix() -> np.ndarray:
    return np.array([VALENCE_AROUSAL[c] for c in CANON])
