"""Extract prosody features (per clip) and face crops (4 fps, per clip) from CREMA-D, in parallel.

    python scripts/real_cremad_extract.py --root /path/to/cremad --out artifacts/cremad
Writes audio.npz (X, quality, duration, file) and faces.npz (crops, clip index, time, quality, sampled-frame counts).
"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emotion_edge.realdata import cremad as C

ROOT = None


def _audio(f):
    try:
        x, q, d = C.audio_features(f"{ROOT}/AudioWAV/{f}.wav")
        return f, x, q, d
    except Exception as e:
        return f, None, 0.0, 0.0


def _face(f):
    try:
        return (f,) + C.face_crops(f"{ROOT}/VideoFlash/{f}.flv")
    except Exception:
        return f, np.zeros((0, 48, 48), np.uint8), np.zeros(0), np.zeros(0), 0


def init(root):
    global ROOT
    ROOT = root


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="artifacts/cremad")
    ap.add_argument("--what", choices=["audio", "face", "both"], default="both")
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    df = C.metadata(a.root)
    files = list(df.file)
    with Pool(a.procs, initializer=init, initargs=(a.root,)) as pool:
        if a.what in ("audio", "both"):
            res = pool.map(_audio, files, chunksize=16)
            dim = next(r[1].shape[0] for r in res if r[1] is not None)
            X = np.stack([r[1] if r[1] is not None else np.full(dim, np.nan, np.float32) for r in res])
            np.savez(out / "audio.npz", X=X, quality=np.array([r[2] for r in res]), duration=np.array([r[3] for r in res]),
                     files=np.array(files))
            print("audio", X.shape, "failed", int(np.isnan(X[:, 0]).sum()), flush=True)
        if a.what in ("face", "both"):
            res = pool.map(_face, files, chunksize=8)
            crops = np.concatenate([r[1] for r in res])
            clip = np.concatenate([np.full(len(r[1]), i) for i, r in enumerate(res)])
            np.savez_compressed(out / "faces.npz", crops=crops, clip=clip, time=np.concatenate([r[2] for r in res]),
                                quality=np.concatenate([r[3] for r in res]), sampled=np.array([r[4] for r in res]),
                                files=np.array(files))
            n_face = np.bincount(clip, minlength=len(files))
            print("faces", crops.shape, "clips without any face", int((n_face == 0).sum()), flush=True)
