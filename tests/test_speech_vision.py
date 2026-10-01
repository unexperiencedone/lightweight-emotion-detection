import numpy as np, torch
from emotion_edge.speech import features as F
from emotion_edge.speech.train import fit as sfit, ProsodyMLP
from emotion_edge.vision.train import fit as vfit, detect_and_crop
from emotion_edge.common import onnx_utils as ou

SR = F.SR


def tone(f0, sec=2.0, tempo=3.0, amp=0.5, vib=0.0):
    t = np.arange(int(sec * SR)) / SR
    ph = 2 * np.pi * np.cumsum(f0 * (1 + vib * np.sin(2 * np.pi * 5 * t))) / SR
    env = (0.5 + 0.5 * np.sin(2 * np.pi * tempo * t)) ** 2
    return (amp * env * (np.sin(ph) + 0.3 * np.sin(2 * ph))).astype(np.float32)


def feat(y, name):
    v, names = F.prosody_features(y)
    return v[names.index(name)]


def test_pitch_features_track_known_f0():
    low, high = tone(120), tone(240)
    st_low, st_high = feat(low, "f0st_mean"), feat(high, "f0st_mean")
    assert abs((st_high - st_low) - 12) < 1.5          # one octave = 12 semitones


def test_tempo_and_loudness_features():
    assert feat(tone(150, tempo=6), "onset_rate") > feat(tone(150, tempo=1.5), "onset_rate")
    assert feat(tone(150, amp=0.5), "logrms_std") > 0


def test_speech_model_roundtrip_and_int8():
    rng = np.random.default_rng(0)
    classes = [dict(f0=110, tempo=1.5), dict(f0=250, tempo=6), dict(f0=180, tempo=3)]
    X, y = [], []
    for c, p in enumerate(classes):
        for _ in range(25):
            X.append(F.prosody_features(tone(p["f0"] * rng.uniform(.9, 1.1), tempo=p["tempo"] * rng.uniform(.9, 1.1)))[0]); y.append(c)
    X, y = np.stack(X), np.array(y)
    idx = rng.permutation(len(y)); tr, va = idx[:50], idx[50:]
    m, acc = sfit(X[tr], y[tr], X[va], y[va], epochs=60, n_out=3)
    assert acc > 0.9
    ou.export_onnx(m, (torch.randn(1, X.shape[1]),), "/tmp/_p.onnx", ["feat"], ["logits"], {"feat": {0: "b"}, "logits": {0: "b"}})
    ou.quantize_dynamic_int8("/tmp/_p.onnx", "/tmp/_p8.onnx")
    ref = m(torch.from_numpy(X[va])).argmax(1).numpy()
    q = ou.ort_session("/tmp/_p8.onnx").run(None, {"feat": X[va]})[0].argmax(1)
    assert (ref == q).mean() > 0.9


def test_face_model_roundtrip_and_static_int8():
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:48, :48]
    def img(c):  # class = grating orientation (7 classes) -- proves the train/export/quantize path, not FER accuracy
        a = c * np.pi / 7
        g = 0.5 + 0.4 * np.sin((xx * np.cos(a) + yy * np.sin(a)) * 0.8 + rng.uniform(0, 6))
        return (np.clip(g + rng.normal(0, .05, g.shape), 0, 1) * 255).astype(np.uint8)
    y = np.repeat(np.arange(7), 40); X = np.stack([img(c) for c in y])
    idx = rng.permutation(len(y)); tr, va = idx[:220], idx[220:]
    m, acc = vfit(X[tr], y[tr], X[va], y[va], epochs=25, aug=False)  # flip would turn mirrored gratings into other classes
    assert acc > 0.6
    f = lambda a: torch.from_numpy(a).float()[:, None] / 255
    ou.export_onnx(m, (torch.rand(1, 1, 48, 48),), "/tmp/_f.onnx", ["img"], ["logits"], {"img": {0: "b"}, "logits": {0: "b"}})
    ou.quantize_static_int8("/tmp/_f.onnx", "/tmp/_f8.onnx", [f(X[i:i + 1]).numpy() for i in range(40)], "img")
    ref = m(f(X[va])).argmax(1).numpy()
    q = ou.ort_session("/tmp/_f8.onnx").run(None, {"img": f(X[va]).numpy()})[0].argmax(1)
    assert (ref == q).mean() > 0.8
    assert sum(p.numel() for p in m.parameters()) < 100_000


def test_detector_returns_none_without_face():
    crop, q = detect_and_crop(np.zeros((200, 200, 3), np.uint8))
    assert crop is None and q == 0.0
