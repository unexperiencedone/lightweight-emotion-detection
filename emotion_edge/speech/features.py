"""Voice-prosody feature extraction (edge-friendly: no neural encoder, ~10 ms of compute per second of audio).

Why hand-crafted prosody and not wav2vec2/HuBERT: those encoders are 95M+ params and dominate the edge budget; the
*paralinguistic* cues that carry affect -- pitch level/range/dynamics, loudness, speaking tempo, voicing, spectral
tilt/timbre -- are captured by 88 summary features, and a ~29k-parameter MLP on top is int8-quantizable.
"""
from __future__ import annotations
import numpy as np
import librosa

SR = 16000
FRAME, HOP = 512, 160          # 32 ms window, 10 ms hop


def _stats(x, prefix):
    x = np.asarray(x, float)
    if x.size == 0:
        x = np.zeros(1)
    t = np.arange(x.size)
    slope = np.polyfit(t, x, 1)[0] if x.size > 1 else 0.0
    return {f"{prefix}_mean": x.mean(), f"{prefix}_std": x.std(), f"{prefix}_min": x.min(), f"{prefix}_max": x.max(),
            f"{prefix}_range": np.ptp(x), f"{prefix}_p10": np.percentile(x, 10), f"{prefix}_p90": np.percentile(x, 90),
            f"{prefix}_slope": slope}


def load_audio(path, sr=SR, max_sec=8.0):
    y, _ = librosa.load(path, sr=sr, mono=True, duration=max_sec)
    return y


def audio_quality(y, sr=SR):
    """Crude SNR estimate (dB, 10th vs 90th percentile frame energy) -> quality in [0,1] for the fusion discount."""
    rms = librosa.feature.rms(y=y, frame_length=FRAME, hop_length=HOP)[0] + 1e-9
    snr = 20 * np.log10(np.percentile(rms, 90) / np.percentile(rms, 10))
    voiced_frac = float((rms > 0.1 * rms.max()).mean())
    return float(np.clip(snr / 30.0, 0, 1) * np.clip(voiced_frac * 2, 0, 1))


def prosody_features(y: np.ndarray, sr: int = SR) -> tuple[np.ndarray, list[str]]:
    y = librosa.util.normalize(y) if y.size and np.abs(y).max() > 0 else y
    f: dict[str, float] = {}
    # pitch: fast YIN (pYIN is ~10x slower; unnecessary for summary stats)
    f0 = librosa.yin(y, fmin=70, fmax=400, sr=sr, frame_length=1024, hop_length=HOP)
    rms = librosa.feature.rms(y=y, frame_length=FRAME, hop_length=HOP)[0]
    n = min(len(f0), len(rms))
    f0, rms = f0[:n], rms[:n]
    voiced = rms > 0.1 * (rms.max() + 1e-9)
    f0v = f0[voiced] if voiced.any() else np.array([0.0])
    semitone = 12 * np.log2(np.clip(f0v, 1, None) / 100.0)      # speaker-agnostic scale
    f.update(_stats(semitone, "f0st"))
    f.update(_stats(np.diff(semitone) if semitone.size > 1 else [0.0], "f0st_delta"))
    f.update(_stats(np.log(rms + 1e-6), "logrms"))
    f.update(_stats(np.diff(np.log(rms + 1e-6)), "logrms_delta"))
    f["voiced_ratio"] = float(voiced.mean())
    # jitter / shimmer proxies (relative cycle-to-cycle variation among voiced frames)
    f["jitter"] = float(np.mean(np.abs(np.diff(f0v)) / (f0v[:-1] + 1e-6))) if f0v.size > 1 else 0.0
    f["shimmer"] = float(np.mean(np.abs(np.diff(rms[voiced])) / (rms[voiced][:-1] + 1e-6))) if voiced.sum() > 1 else 0.0
    # tempo: syllable-nucleus proxy = energy-onset rate; pause ratio
    on = librosa.onset.onset_detect(y=y, sr=sr, hop_length=HOP, units="time")
    dur = len(y) / sr
    f["onset_rate"] = len(on) / max(dur, 1e-3)
    f["pause_ratio"] = float((rms < 0.05 * (rms.max() + 1e-9)).mean())
    f["duration"] = dur
    # timbre / spectral
    S = np.abs(librosa.stft(y, n_fft=FRAME, hop_length=HOP))
    f.update(_stats(librosa.feature.spectral_centroid(S=S, sr=sr)[0] / 1000, "centroid"))
    f.update(_stats(librosa.feature.spectral_rolloff(S=S, sr=sr)[0] / 1000, "rolloff"))
    f.update(_stats(librosa.feature.zero_crossing_rate(y, frame_length=FRAME, hop_length=HOP)[0], "zcr"))
    mf = librosa.feature.mfcc(S=librosa.power_to_db(librosa.feature.melspectrogram(S=S ** 2, sr=sr, n_mels=40)), n_mfcc=13)
    for i in range(13):
        f[f"mfcc{i}_mean"], f[f"mfcc{i}_std"] = mf[i].mean(), mf[i].std()
    names = sorted(f)
    return np.array([f[k] for k in names], dtype=np.float32), names
