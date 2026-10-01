"""Pretrained speech-encoder features for CREMA-D: every hidden layer, mean- and std-pooled over time.

    python scripts/ssl_extract.py --root <cremad> --model ntu-spml/distilhubert --out <feats>/ssl_distilhubert.npz

The encoder is frozen; a small head (scripts/upgrade_models.py) learns a softmax-weighted sum of layers
(SUPERB recipe, Yang et al., 2021). Stored as float16: (clips, layers, 2*hidden).
"""
import argparse
import sys
import time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emotion_edge.realdata import cremad as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-sec", type=float, default=6.0)
    a = ap.parse_args()
    import librosa
    from transformers import AutoModel
    torch.set_num_threads(4)
    m = AutoModel.from_pretrained(a.model).eval()
    df = C.metadata(a.root)
    feats = None
    t0 = time.time()
    for i, f in enumerate(df.file):
        y, _ = librosa.load(f"{a.root}/AudioWAV/{f}.wav", sr=16000, duration=a.max_sec)
        x = torch.from_numpy((y - y.mean()) / (y.std() + 1e-7)).float()[None]
        with torch.no_grad():
            hs = m(x, output_hidden_states=True).hidden_states
        H = torch.stack(hs, 0)[:, 0]                                   # (layers, T, hidden)
        v = torch.cat([H.mean(1), H.std(1)], -1).numpy().astype(np.float16)
        if feats is None:
            feats = np.zeros((len(df),) + v.shape, np.float16)
        feats[i] = v
        if i % 1000 == 0:
            print(f"{i}/{len(df)} {time.time() - t0:.0f}s", flush=True)
    np.savez(a.out, X=feats, files=np.array(df.file, dtype=str), model=a.model)
    print("done", feats.shape, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
