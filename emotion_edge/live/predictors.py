"""Model wrappers used by the live pipeline. Each returns *logits in its own label space* plus a quality score;
temperatures for calibration are read from the training run's results.json when available."""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np

from emotion_edge.common.onnx_utils import ort_session


def _temperature(results_json):
    try:
        return float(json.loads(Path(results_json).read_text())["calibration"]["temperature"])
    except Exception:
        return 1.0


class TextPredictor:
    """artifacts/text/<model>.onnx + tokenizer dir (+ vocab_remap.npy for the pruned variants)."""

    def __init__(self, onnx_path, tokenizer_dir, remap_path=None, results_json=None, max_len=64, threads=1):
        from transformers import AutoTokenizer
        self.sess = ort_session(onnx_path, threads)
        self.tok = AutoTokenizer.from_pretrained(tokenizer_dir)
        self.remap = np.load(remap_path) if remap_path and Path(remap_path).exists() else None
        self.max_len = max_len
        self.temperature = _temperature(results_json) if results_json else 1.0

    def __call__(self, text: str):
        t0 = time.perf_counter()
        enc = self.tok([text], truncation=True, max_length=self.max_len, return_tensors="np")
        ids = enc["input_ids"].astype(np.int64)
        if self.remap is not None:
            ids = self.remap[ids]
        logits = self.sess.run(None, {"input_ids": ids, "attention_mask": enc["attention_mask"].astype(np.int64)})[0][0]
        return logits, (time.perf_counter() - t0) * 1000


class SpeechPredictor:
    def __init__(self, onnx_path, results_json=None, threads=1):
        self.sess = ort_session(onnx_path, threads)
        self.dim = self.sess.get_inputs()[0].shape[1]
        self.temperature = _temperature(results_json) if results_json else 1.0

    def __call__(self, samples: np.ndarray, sr=16000):
        from emotion_edge.speech.features import prosody_features, audio_quality
        t0 = time.perf_counter()
        f, _ = prosody_features(samples, sr)
        if isinstance(self.dim, int) and f.shape[0] != self.dim:
            raise ValueError(f"prosody model expects {self.dim} features, extractor produced {f.shape[0]}")
        logits = self.sess.run(None, {"feat": f[None]})[0][0]
        return logits, audio_quality(samples, sr), (time.perf_counter() - t0) * 1000


class FacePredictor:
    def __init__(self, onnx_path, results_json=None, threads=1):
        self.sess = ort_session(onnx_path, threads)
        self.temperature = _temperature(results_json) if results_json else 1.0

    def __call__(self, bgr: np.ndarray):
        from emotion_edge.vision.train import detect_and_crop
        t0 = time.perf_counter()
        crop, q = detect_and_crop(bgr)
        if crop is None:
            return None, 0.0, (time.perf_counter() - t0) * 1000
        logits = self.sess.run(None, {"img": crop[None, None].astype(np.float32)})[0][0]
        return logits, q, (time.perf_counter() - t0) * 1000
