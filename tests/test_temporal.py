import json
import numpy as np
import pytest
from emotion_edge.labels import CANON, CIDX
from emotion_edge.temporal.segment import TextSegmenter, AudioSegmenter, FrameSampler
from emotion_edge.temporal.tracker import StickyFilter, MoodWindow, Obs, behaviour, fuse_moods
from emotion_edge.temporal.session import LiveSession

K = len(CANON)


def peaked(c, p=0.9):
    q = np.full(K, (1 - p) / (K - 1)); q[CIDX[c]] = p
    return q


# ---------------------------------------------------------------- segmentation
def test_text_closes_on_sentence_end_pause_and_cap():
    s = TextSegmenter()
    out = []
    for i, w in enumerate("i really thought this would work.".split()):
        out += s.push_word(w, 0.3 * i)
    assert len(out) == 1 and out[0].reason == "sentence_end"
    out = s.push_word("ok", 5.0) + s.push_word("so", 5.2) + s.push_word("anyway", 5.4) + s.push_word("then", 7.0)
    assert out and out[0].reason.startswith("pause") and out[0].text == "ok so anyway"
    s2 = TextSegmenter(max_words=5)
    blocks = sum((s2.push_word(f"w{i}", 0.1 * i) for i in range(12)), [])
    assert [b.n_words for b in blocks] == [5, 5]


def test_short_fragment_is_merged_not_emitted():
    s = TextSegmenter()
    out = s.push_word("ok.", 0.0) + s.push_word("that", 1.5) + s.push_word("is", 1.7) + s.push_word("fine.", 1.9)
    assert len(out) == 1 and out[0].text == "ok. that is fine."
    assert s.tick(10) == []


def test_speaker_change_closes_block():
    s = TextSegmenter()
    s.push_word("hello", 0.0, "A"); s.push_word("there", 0.2, "A")
    out = s.push_word("hi", 0.4, "B")
    assert out and out[0].reason == "speaker_change" and out[0].speaker == "A"


def _bursts(spec, sr=16000, seed=0):
    rng = np.random.default_rng(seed)
    y = []
    for dur, voiced in spec:
        n = int(dur * sr); t = np.arange(n) / sr
        y.append(0.003 * rng.normal(size=n) + voiced * 0.3 * np.sin(2 * np.pi * 150 * t))
    return np.concatenate(y).astype(np.float32)


def test_audio_vad_segments():
    y = _bursts([(0.5, 0), (2.0, 1), (0.2, 0), (1.0, 1), (1.0, 0), (0.3, 1), (1.0, 0), (8.0, 1), (1.0, 0)])
    seg = AudioSegmenter()
    blocks = sum((seg.push(y[i:i + 1600]) for i in range(0, len(y), 1600)), []) + seg.flush()
    durs = [round(b.t_end - b.t_start, 1) for b in blocks]
    assert abs(durs[0] - 3.2) < 0.15            # 0.2 s gap kept inside the segment
    assert all(d >= 0.8 for d in durs)           # 0.3 s blip dropped
    assert any(b.reason == "max_len" for b in blocks) and max(durs) <= 6.0 + 1e-6


def test_frame_sampler_rate():
    fs = FrameSampler(fps=4)
    assert sum(fs.accept(i / 30) for i in range(300)) == 40


# ---------------------------------------------------------------- filter
def test_filter_smooths_and_decays_to_base():
    f = StickyFilter(dwell_s=4)
    for t in np.arange(0, 5, 0.25):
        f.update(t, peaked("joy"), 1.0)
    assert f.b.argmax() == CIDX["joy"] and f.b.max() > 0.95
    f.update(5.0, peaked("anger"), 1.0)            # one outlier frame does not flip the state
    assert f.b.argmax() == CIDX["joy"]
    f.predict(60.0)                                 # long silence -> back to base rate
    assert f.informativeness < 0.01


def test_low_quality_has_less_influence():
    a, b = StickyFilter(), StickyFilter()
    a.update(0, peaked("fear"), 1.0); b.update(0, peaked("fear"), 0.1)
    assert a.b[CIDX["fear"]] > b.b[CIDX["fear"]]


# ---------------------------------------------------------------- mood
def test_mood_dominant_mixed_insufficient_transition():
    w = MoodWindow(window_s=30, tau_c=1.0)
    for t in range(30):
        w.add(Obs(float(t), peaked("sadness"), 1.0, 1.0))
    e = w.estimate(30.0)
    assert e.label == "sadness" and e.state == "dominant" and e.p_dominant.max() > 0.95
    lo, hi = e.ci90[CIDX["sadness"]]
    assert lo < e.mean[CIDX["sadness"]] < hi

    w = MoodWindow()
    for t in range(30):
        w.add(Obs(float(t), peaked("joy" if t % 2 else "love"), 1.0, 1.0))
    assert w.estimate(30.0).state == "mixed"

    w = MoodWindow()
    w.add(Obs(29.0, peaked("anger"), 1.0, 1.0))
    assert w.estimate(30.0).state == "insufficient"

    w = MoodWindow()
    for t in range(30):
        w.add(Obs(float(t), peaked("neutral" if t < 15 else "joy"), 1.0, 1.0))
    e = w.estimate(30.0)
    assert e.state == "transition" and e.transition == ("neutral", "joy")


def test_correlated_frames_do_not_inflate_evidence():
    fast, slow = MoodWindow(tau_c=1.0), MoodWindow(tau_c=1.0)
    for t in np.arange(0, 10, 0.1):               # 10 fps for 10 s
        fast.add(Obs(float(t), peaked("joy"), 1.0, 0.1))
    for t in range(10):                            # 1 fps for 10 s
        slow.add(Obs(float(t), peaked("joy"), 1.0, 1.0))
    assert abs(fast.estimate(10.0).n_eff - slow.estimate(10.0).n_eff) < 0.6


def test_fused_mood_adds_evidence_and_reports_disagreement():
    a, b = MoodWindow(), MoodWindow()
    for t in range(20):
        a.add(Obs(float(t), peaked("joy"), 1.0, 1.0)); b.add(Obs(float(t), peaked("anger"), 1.0, 1.0))
    est = fuse_moods({"text": a, "face": b}, 20.0)
    assert est.n_eff == pytest.approx(a.estimate(20.0).n_eff * 2, rel=1e-6)
    assert est.state == "conflicted" and est.extra["cross_modal_jsd"] > 0.3


# ---------------------------------------------------------------- behaviour
def test_behaviour_tags():
    t = np.arange(0, 30, 0.5)
    stable = np.tile(peaked("sadness"), (len(t), 1))
    assert behaviour(t, stable)["tags"] == ["stable"]
    w = np.linspace(0, 1, len(t))[:, None]
    esc = (1 - w) * peaked("sadness") + w * peaked("fear")
    assert "escalating" in behaviour(t, esc)["tags"]
    vol = np.array([peaked("joy") if (i // 2) % 2 else peaked("anger") for i in range(len(t))])
    assert "volatile" in behaviour(t, vol)["tags"]
    assert behaviour(t[:3], stable[:3])["tags"] == ["insufficient"]


# ---------------------------------------------------------------- session
def test_session_conflict_and_staleness():
    s = LiveSession()
    for t in np.arange(0, 10, 0.25):
        s.observe("face", t, probs=np.eye(7)[0] * 0.9 + 0.1 / 7, quality=1.0, dur=0.25)          # angry face
        if t % 2 == 0:
            s.observe("text", t, probs=np.eye(6)[1] * 0.9 + 0.1 / 6, quality=1.0)               # joyful words
    assert s.tick(10.0)["state"] == "conflict"
    s.observe("face", 10.0, probs=np.eye(7)[0] * 0.9 + 0.1 / 7)
    r = s.tick(40.0)                                # text silent for 30 s -> decays, face (also stale) - no conflict
    assert r["state"] != "conflict"


def test_demo_pipeline_runs_and_beats_static_fusion():
    from emotion_edge.temporal.evaluate import evaluate_seed
    r = evaluate_seed(3)
    assert r["temporal"]["acc_clean"] > r["baseline"]["acc_clean"]
    assert r["temporal"]["flicker_per_min"] < r["baseline"]["flicker_per_min"] / 5
    assert r["temporal"]["conflict_rate_masking"] > r["baseline"]["conflict_rate_masking"]
