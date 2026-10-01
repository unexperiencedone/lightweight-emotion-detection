"""End-to-end live pipeline.

  # 1) demo: scripted synthetic session (no models or data needed) -> JSONL + all temporal figures + evaluation
  python scripts/live.py demo --out results/live_demo

  # 2) replay recorded files in timestamp order (deterministic; same code path as live)
  python scripts/live.py files --video clip.mp4 --audio clip.wav --transcript clip.jsonl --out results/live_files \
      --text-onnx artifacts/text/model_pruned_int8_emb8.onnx --tokenizer artifacts/text/torch_fp32 \
      --remap artifacts/text/vocab_remap.npy --text-results artifacts/text/results.json \
      --speech-onnx artifacts/speech/prosody_int8.onnx --face-onnx artifacts/face/face_int8.onnx

  # 3) real time from webcam + microphone (needs `pip install sounddevice` for audio); typed text via --transcript
  python scripts/live.py realtime --seconds 60 --out results/live_rt  [model flags as above]

Every mode prints instant readings (every tick) and window reports (every hop) and writes records.jsonl.
Any modality whose model flag is omitted is simply not used.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emotion_edge.live.pipeline import LivePipeline
from emotion_edge.live import sources as SRC
from emotion_edge.temporal.session import SessionConfig


def build_predictors(a):
    from emotion_edge.live.predictors import TextPredictor, SpeechPredictor, FacePredictor
    text = TextPredictor(a.text_onnx, a.tokenizer, a.remap, a.text_results) if a.text_onnx else None
    speech = SpeechPredictor(a.speech_onnx, a.speech_results) if a.speech_onnx else None
    face = FacePredictor(a.face_onnx, a.face_results) if a.face_onnx else None
    return text, speech, face


def printer(verbose):
    def sink(r):
        if r["type"] == "instant" and verbose and r.get("label") and abs(r["t"] - round(r["t"])) < 1e-6:
            print(f"[{r['t']:7.1f}s] instant {r['label']:<8} p={r['p_top1']:.2f} {r['state']:<10} {r.get('explanation', '')}")
        elif r["type"] == "window" and r.get("fused_mood"):
            fm = r["fused_mood"]
            tr = f" {fm['transition'][0]}->{fm['transition'][1]}" if fm.get("transition") else ""
            print(f"[{r['t']:7.1f}s] WINDOW  mood {fm['label']:<8} {fm['state']}{tr}  P(dominant)={fm['p_dominant']:.2f}"
                  f"  pattern={','.join(r['behaviour']['fused']['tags'])}  conflict_share={r['conflict_share']:.2f}")
    return sink


def write(recs, out):
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "records.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r, default=float) + "\n")
    print(f"wrote {len(recs)} records to {out / 'records.jsonl'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["demo", "files", "realtime"])
    ap.add_argument("--out", default="results/live")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--window", type=float, default=30.0)
    ap.add_argument("--hop", type=float, default=5.0)
    ap.add_argument("--tick", type=float, default=0.5)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--video"); ap.add_argument("--audio"); ap.add_argument("--transcript")
    ap.add_argument("--camera", type=int, default=0); ap.add_argument("--no-mic", action="store_true")
    ap.add_argument("--seconds", type=float, default=60.0)
    for m in ("text", "speech", "face"):
        ap.add_argument(f"--{m}-onnx"); ap.add_argument(f"--{m}-results")
    ap.add_argument("--tokenizer"); ap.add_argument("--remap")
    a = ap.parse_args()
    out = Path(a.out)
    cfg = SessionConfig(window_s=a.window, hop_s=a.hop, tick_s=a.tick)

    if a.mode == "demo":
        from emotion_edge.live import scenario as S
        from emotion_edge.temporal import viz
        from emotion_edge.temporal.evaluate import evaluate
        p = LivePipeline(cfg, sink=printer(not a.quiet))
        recs = p.run(S.events(a.seed))
        write(recs, out)
        figs = Path("docs/figures")
        figs.mkdir(parents=True, exist_ok=True)
        viz.fig_timeline(recs, S.PHASES, figs / "live_timeline.png")
        viz.fig_mood(recs, figs / "live_mood.png")
        viz.fig_va(recs, S.PHASES, figs / "live_valence_arousal.png")
        viz.fig_filter_effect(recs, figs / "live_filter_effect.png")
        viz.fig_segmentation(figs / "segmentation.png")
        w = [r for r in recs if r["type"] == "window" and abs(r["t"] - 60) < 1e-6][0]
        viz.fig_dirichlet(w, figs / "mood_dirichlet_t60.png")
        res = evaluate(range(10))
        (out / "temporal_eval.json").write_text(json.dumps(res, indent=2))
        Path("results").mkdir(exist_ok=True)
        Path("results/temporal_eval.json").write_text(json.dumps(res, indent=2))
        viz.fig_eval(res, figs / "temporal_eval.png")
        print("figures written to docs/figures/; evaluation in results/temporal_eval.json")
        return

    text, speech, face = build_predictors(a)
    p = LivePipeline(cfg, text=text, speech=speech, face=face, sink=printer(not a.quiet))
    if a.mode == "files":
        streams = []
        if a.video: streams.append(SRC.video_events(a.video))
        if a.audio: streams.append(SRC.wav_events(a.audio))
        if a.transcript: streams.append(SRC.transcript_events(a.transcript))
        if not streams:
            sys.exit("give at least one of --video / --audio / --transcript")
        recs = p.run(SRC.merge(*streams))
    else:
        recs = p.run(SRC.realtime_events(camera=a.camera, mic=not a.no_mic, duration_s=a.seconds))
    write(recs, out)
    if a.mode == "files" and not a.quiet:
        from emotion_edge.temporal import viz
        viz.fig_timeline(recs, [], out / "timeline.png", title="Session timeline (files)")
        if any(r["type"] == "window" and r.get("fused_mood") for r in recs):
            viz.fig_mood(recs, out / "mood.png")


if __name__ == "__main__":
    main()
