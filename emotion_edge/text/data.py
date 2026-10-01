"""dair-ai/emotion loading (Hub or local files) and a labelled synthetic stand-in for smoke tests."""
from __future__ import annotations
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
from emotion_edge.labels import TEXT_LABELS


def load_dair(data_dir: str | None = None) -> dict[str, pd.DataFrame]:
    """Return {'train','validation','test'} DataFrames with columns text,label.

    data_dir may hold {train,validation,test}.parquet|.jsonl|.csv (offline use);
    otherwise the dataset is pulled from the Hub (`dair-ai/emotion`, config `split`: 16k/2k/2k).
    """
    out = {}
    if data_dir:
        d = Path(data_dir)
        for s in ("train", "validation", "test"):
            for ext, rd in ((".parquet", pd.read_parquet), (".jsonl", lambda p: pd.read_json(p, lines=True)),
                            (".csv", pd.read_csv)):
                if (d / f"{s}{ext}").exists():
                    out[s] = rd(d / f"{s}{ext}")[["text", "label"]]
                    break
            else:
                raise FileNotFoundError(f"{s}.(parquet|jsonl|csv) not found in {d}")
        return out
    from datasets import load_dataset
    ds = load_dataset("dair-ai/emotion", "split")
    return {s: ds[s].to_pandas()[["text", "label"]] for s in ("train", "validation", "test")}


# ---- synthetic stand-in (SMOKE TESTS ONLY; never used for reported accuracy) -------------------
_KW = {
    "sadness": ["miserable", "lonely", "heartbroken", "gloomy", "crying", "hopeless"],
    "joy": ["delighted", "thrilled", "cheerful", "wonderful", "happy", "excited"],
    "love": ["adore", "cherish", "affectionate", "devoted", "sweet", "caring"],
    "anger": ["furious", "outraged", "irritated", "hostile", "resentful", "enraged"],
    "fear": ["terrified", "anxious", "panicked", "frightened", "nervous", "dread"],
    "surprise": ["astonished", "stunned", "shocked", "amazed", "unexpected", "startled"],
}
_FILL = ["i", "feel", "so", "today", "and", "really", "because", "of", "this", "the", "my", "all", "day"]


def synthetic_dair(n_train=1200, n_val=200, n_test=200, seed=0):
    rng = random.Random(seed)

    def make(n):
        rows = []
        for _ in range(n):
            y = rng.randrange(6)
            kw = rng.sample(_KW[TEXT_LABELS[y]], 2)
            toks = rng.choices(_FILL, k=rng.randint(4, 12)) + kw
            rng.shuffle(toks)
            rows.append(("i feel " + " ".join(toks), y))
        return pd.DataFrame(rows, columns=["text", "label"])

    return {"train": make(n_train), "validation": make(n_val), "test": make(n_test)}


def write_synthetic_vocab(path: str, frames: dict) -> str:
    words = {w for df in frames.values() for t in df.text for w in t.split()}
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + sorted(words)
    Path(path).write_text("\n".join(vocab))
    return path
