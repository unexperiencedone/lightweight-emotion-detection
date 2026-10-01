"""End-to-end latency budget for one multimodal decision on a single pinned core.
Models use RANDOM weights (architecture-level timing only). Stages: text tokenise+infer, audio prosody extraction+infer,
face detect+crop+infer, fusion.   python scripts/multimodal_latency.py results/multimodal_latency.json"""
import json, os, sys, time, numpy as np, torch
os.sched_setaffinity(0, {sorted(os.sched_getaffinity(0))[0]})
torch.set_num_threads(1)
from emotion_edge.common import onnx_utils as ou
from emotion_edge.speech import features as F
from emotion_edge.speech.train import ProsodyMLP
from emotion_edge.vision.model import MiniXception
from emotion_edge.vision.train import detect_and_crop
from emotion_edge.fusion.fuse import Modality, fuse
from emotion_edge.labels import TEXT_LABELS

def bench(fn, n=50, warm=5):
    for _ in range(warm): fn()
    t = []
    for _ in range(n):
        s = time.perf_counter(); fn(); t.append((time.perf_counter() - s) * 1000)
    return {"p50_ms": float(np.percentile(t, 50)), "p95_ms": float(np.percentile(t, 95))}

out = {}
os.makedirs("artifacts/mm", exist_ok=True)
y = (0.3 * np.sin(2 * np.pi * 180 * np.arange(3 * F.SR) / F.SR)).astype(np.float32)          # 3 s clip
feat, names = F.prosody_features(y)
sp = ProsodyMLP(len(names)).eval()
ou.export_onnx(sp, (torch.randn(1, len(names)),), "artifacts/mm/sp.onnx", ["feat"], ["logits"], {"feat": {0: "b"}, "logits": {0: "b"}})
ou.quantize_dynamic_int8("artifacts/mm/sp.onnx", "artifacts/mm/sp8.onnx")
fc = MiniXception().eval()
ou.export_onnx(fc, (torch.rand(1, 1, 48, 48),), "artifacts/mm/fc.onnx", ["img"], ["logits"], {"img": {0: "b"}, "logits": {0: "b"}})
ou.quantize_static_int8("artifacts/mm/fc.onnx", "artifacts/mm/fc8.onnx", [np.random.rand(1, 1, 48, 48).astype(np.float32) for _ in range(20)], "img")
s_sp, s_fc = ou.ort_session("artifacts/mm/sp8.onnx"), ou.ort_session("artifacts/mm/fc8.onnx")
img = (np.random.rand(240, 320, 3) * 255).astype(np.uint8)

out["audio_prosody_extract_3s"] = bench(lambda: F.prosody_features(y), 20, 3)
out["audio_infer_int8"] = bench(lambda: s_sp.run(None, {"feat": feat[None]}))
out["face_detect_crop_320x240"] = bench(lambda: detect_and_crop(img))
x = np.random.rand(1, 1, 48, 48).astype(np.float32)
out["face_infer_int8"] = bench(lambda: s_fc.run(None, {"img": x}))
rng = np.random.default_rng(0)
mods = [Modality("text", rng.normal(size=6)), Modality("speech", rng.normal(size=8)), Modality("face", rng.normal(size=7))]
out["fusion"] = bench(lambda: fuse(mods), 500)
out["sizes_mb"] = {"prosody_int8": ou.file_mb("artifacts/mm/sp8.onnx"), "face_int8": ou.file_mb("artifacts/mm/fc8.onnx")}
out["params"] = {"prosody_mlp": sum(p.numel() for p in sp.parameters()), "face_cnn": sum(p.numel() for p in fc.parameters())}
out["note"] = "random weights; 1 pinned host core; add the text figure from results/edge_text_*.json"
json.dump(out, open(sys.argv[1], "w"), indent=2)
for k, v in out.items():
    if isinstance(v, dict) and "p50_ms" in v: print(f"{k:28s} p50 {v['p50_ms']:.2f} ms  p95 {v['p95_ms']:.2f} ms")
print(out["sizes_mb"], out["params"])
