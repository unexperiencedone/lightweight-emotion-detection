"""CREMA-D (Cao et al., 2014; ODbL): 7,442 clips, 91 actors, 12 sentences, 6 emotions, audio + video.

Besides the *intended* (acted) label, every clip has crowd ratings collected separately for
voice-only (row prefix 1), face-only (2) and audio-visual (3) presentation (~9 raters each).
These give us, for real recordings:
  * human perception distributions per modality      -> calibration targets
  * human disagreement within a modality              -> ground truth for "ambiguous"
  * human voice-only vs face-only disagreement        -> ground truth for "incongruent / conflict"
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from emotion_edge.labels import SPEECH_LABELS, FACE_LABELS, CANON, CIDX

CODES = ["A", "D", "F", "H", "N", "S"]                           # CREMA-D vote columns
NAMES = ["anger", "disgust", "fear", "joy", "neutral", "sadness"]  # canonical names of the 6 CREMA-D classes
FILE_CODE = {"ANG": "A", "DIS": "D", "FEA": "F", "HAP": "H", "NEU": "N", "SAD": "S"}
TO_SPEECH = {"A": "angry", "D": "disgust", "F": "fearful", "H": "happy", "N": "neutral", "S": "sad"}
TO_FACE = {"A": "angry", "D": "disgust", "F": "fear", "H": "happy", "N": "neutral", "S": "sad"}


def metadata(root) -> pd.DataFrame:
    root = Path(root)
    s = pd.read_csv(root / "summaryTable.csv")
    df = pd.DataFrame({"file": s.FileName})
    parts = df.file.str.split("_")
    df["actor"] = parts.str[0].astype(int)
    df["sentence"] = parts.str[1]
    df["code"] = parts.str[2].map(FILE_CODE)
    df["intensity"] = parts.str[3]
    df["y"] = df.code.map(CODES.index)                           # 0..5 in CREMA order
    t = pd.read_csv(root / "tabulatedVotes.csv")
    pref = t.iloc[:, 0].astype(str).str[0].map({"1": "voice", "2": "face", "3": "av"})
    for mod in ("voice", "face", "av"):
        x = t[pref == mod].set_index("fileName").loc[df.file, CODES].to_numpy(float)
        df[f"h_{mod}"] = list(x / x.sum(1, keepdims=True))        # human vote distribution over the 6 classes
        df[f"h_{mod}_agree"] = x.max(1) / x.sum(1)                 # share of raters on the top emotion
        df[f"h_{mod}_top"] = x.argmax(1)
    demo = pd.read_csv(root / "VideoDemographics.csv").set_index("ActorID")
    df["sex"] = df.actor.map(demo.Sex)
    return df


def actor_split(df, seed=0, n_val=15, n_test=15):
    actors = np.array(sorted(df.actor.unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(actors)
    test, val = set(actors[:n_test]), set(actors[n_test:n_test + n_val])
    return np.where(df.actor.isin(test), "test", np.where(df.actor.isin(val), "val", "train"))


def speech_index(codes):
    return np.array([SPEECH_LABELS.index(TO_SPEECH[c]) for c in codes])


def face_index(codes):
    return np.array([FACE_LABELS.index(TO_FACE[c]) for c in codes])


def canon_to_crema(p: np.ndarray) -> np.ndarray:
    """Canonical 8-class posterior -> the 6 CREMA-D classes (joy+love -> happy, surprise dropped), renormalised."""
    p = np.atleast_2d(p)
    out = np.stack([p[:, CIDX["anger"]], p[:, CIDX["disgust"]], p[:, CIDX["fear"]],
                    p[:, CIDX["joy"]] + p[:, CIDX["love"]], p[:, CIDX["neutral"]], p[:, CIDX["sadness"]]], 1)
    return out / out.sum(1, keepdims=True)


# --------------------------------------------------------------------------------------------- extraction
def audio_features(path):
    from emotion_edge.speech.features import load_audio, prosody_features, audio_quality
    y = load_audio(path)
    f, names = prosody_features(y)
    return f, audio_quality(y), len(y) / 16000


def face_crops(path, fps=4.0, size=48):
    """Sample frames at `fps` (fixed grid), Haar-detect the largest face, return (crops, times, qualities, n_sampled)."""
    import cv2
    from emotion_edge.vision.train import detect_and_crop
    cap = cv2.VideoCapture(str(path))
    vfps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    crops, times, qs, i, nxt, n = [], [], [], 0, 0.0, 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        t = i / vfps
        if t + 1e-9 >= nxt:
            nxt += 1.0 / fps
            n += 1
            c, q = detect_and_crop(fr, size)
            if c is not None:
                crops.append((c * 255).astype(np.uint8)); times.append(t); qs.append(q)
        i += 1
    cap.release()
    return (np.stack(crops) if crops else np.zeros((0, size, size), np.uint8)), np.array(times), np.array(qs), n
