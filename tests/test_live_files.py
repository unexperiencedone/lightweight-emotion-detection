"""Integration: real file sources + real predictors (tiny RANDOM models) through the live pipeline. Plumbing only."""
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def _make_models(d: Path):
    from transformers import BertTokenizerFast, DistilBertConfig, DistilBertForSequenceClassification
    from emotion_edge.common import onnx_utils as ou
    from emotion_edge.text.model import ExportWrapper
    from emotion_edge.speech.features import prosody_features
    from emotion_edge.speech.train import ProsodyMLP
    from emotion_edge.vision.model import MiniXception
    words = "[PAD] [UNK] [CLS] [SEP] [MASK] i am so happy today this is awful really fine".split()
    tok = BertTokenizerFast(vocab={w: i for i, w in enumerate(words)}, do_lower_case=True)
    tok.save_pretrained(d / "tok")
    m = DistilBertForSequenceClassification(DistilBertConfig(vocab_size=len(words), dim=32, hidden_dim=64, n_layers=1, n_heads=2, num_labels=6)).eval()
    ax = {"input_ids": {0: "b", 1: "s"}, "attention_mask": {0: "b", 1: "s"}, "logits": {0: "b"}}
    ou.export_onnx(ExportWrapper(m), (torch.randint(5, 10, (1, 6)), torch.ones(1, 6, dtype=torch.long)), d / "text.onnx",
                   ["input_ids", "attention_mask"], ["logits"], ax)
    dim = prosody_features(np.random.randn(16000).astype(np.float32) * 0.1)[0].shape[0]
    ou.export_onnx(ProsodyMLP(dim).eval(), (torch.randn(1, dim),), d / "sp.onnx", ["feat"], ["logits"], {"feat": {0: "b"}, "logits": {0: "b"}})
    ou.export_onnx(MiniXception().eval(), (torch.rand(1, 1, 48, 48),), d / "face.onnx", ["img"], ["logits"], {"img": {0: "b"}, "logits": {0: "b"}})


def _make_media(d: Path, secs=20):
    import cv2, soundfile as sf
    sr = 16000
    t = np.arange(secs * sr) / sr
    voiced = ((t % 5) < 3.0).astype(np.float32)               # 3 s talk / 2 s pause
    y = 0.003 * np.random.default_rng(0).normal(size=t.size) + voiced * 0.3 * np.sin(2 * np.pi * 170 * t)
    sf.write(d / "a.wav", y.astype(np.float32), sr)
    vw = cv2.VideoWriter(str(d / "v.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 10, (160, 120))
    for i in range(secs * 10):
        vw.write(np.full((120, 160, 3), (i * 3) % 255, np.uint8))
    vw.release()
    lines = [{"t": 3.1, "text": "i am so happy today"}, {"t": 8.2, "word": "this", "speaker": "A"},
             {"t": 8.4, "word": "is", "speaker": "A"}, {"t": 8.6, "word": "really", "speaker": "A"},
             {"t": 8.8, "word": "awful.", "speaker": "A"}, {"t": 15.0, "text": "fine"}]
    (d / "t.jsonl").write_text("\n".join(json.dumps(l) for l in lines))


def test_files_mode_end_to_end(tmp_path):
    _make_models(tmp_path)
    _make_media(tmp_path)
    out = tmp_path / "out"
    cmd = [sys.executable, str(ROOT / "scripts/live.py"), "files", "--out", str(out), "--quiet", "--window", "10", "--hop", "5",
           "--video", str(tmp_path / "v.avi"), "--audio", str(tmp_path / "a.wav"), "--transcript", str(tmp_path / "t.jsonl"),
           "--text-onnx", str(tmp_path / "text.onnx"), "--tokenizer", str(tmp_path / "tok"),
           "--speech-onnx", str(tmp_path / "sp.onnx"), "--face-onnx", str(tmp_path / "face.onnx")]
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, timeout=600)
    assert p.returncode == 0, p.stderr[-2000:]
    recs = [json.loads(l) for l in open(out / "records.jsonl")]
    blocks = [r for r in recs if r["type"] == "block"]
    text = [b for b in blocks if b["modality"] == "text"]
    speech = [b for b in blocks if b["modality"] == "speech"]
    assert [b["text"] for b in text] == ["i am so happy today", "this is really awful.", "fine"]
    assert text[1]["closed_by"] == "sentence_end"
    assert 3 <= len(speech) <= 5 and all(2.4 <= b["seg"][1] - b["seg"][0] <= 3.3 for b in speech)
    assert not [b for b in blocks if b["modality"] == "face"]      # flat frames: no face detected -> modality absent
    assert any(r["type"] == "window" and r["fused_mood"] for r in recs)
    assert all("proc_ms" in b for b in text + speech)
