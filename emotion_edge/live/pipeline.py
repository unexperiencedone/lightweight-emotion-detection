"""LivePipeline: events -> block segmentation -> unimodal models -> temporal session -> records.

Records (JSON-serialisable dicts, also written to JSONL by scripts/live.py):
  {"type": "block",   ...}  one analysed block per modality (raw, unsmoothed reading + processing time)
  {"type": "instant", ...}  every tick_s: fused instant emotion + ambiguity state + per-modality filtered readings
  {"type": "window",  ...}  every hop_s: per-modality mood, fused mood, behavioural pattern over window_s
"""
from __future__ import annotations
from typing import Callable, Iterable

from emotion_edge.temporal.segment import TextSegmenter, AudioSegmenter, FrameSampler
from emotion_edge.temporal.session import LiveSession, SessionConfig
from emotion_edge.live.sources import Event


class LivePipeline:
    def __init__(self, cfg: SessionConfig | None = None, text=None, speech=None, face=None,
                 text_seg: TextSegmenter | None = None, audio_seg: AudioSegmenter | None = None,
                 sampler: FrameSampler | None = None, sink: Callable[[dict], None] | None = None):
        self.session = LiveSession(cfg)
        self.cfg = self.session.cfg
        self.text, self.speech, self.face = text, speech, face
        for m, p in (("text", text), ("speech", speech), ("face", face)):
            if p is not None:
                self.cfg.temperatures[m] = getattr(p, "temperature", 1.0)
        self.text_seg = text_seg or TextSegmenter()
        self.audio_seg = audio_seg or AudioSegmenter()
        self.sampler = sampler or FrameSampler()
        self.records: list[dict] = []
        self.sink = sink
        self._next_tick = 0.0
        self._next_report = self.cfg.hop_s

    def _emit(self, rec):
        if rec is None:
            return
        self.records.append(rec)
        if self.sink:
            self.sink(rec)

    # -------------------------------------------------------------------------- clock
    def advance(self, now: float):
        for b in self.text_seg.tick(now):
            self._text_block(b)
        while self._next_tick <= now:
            self._emit(self.session.tick(self._next_tick))
            if self._next_report <= self._next_tick:
                self._emit(self.session.report(self._next_report))
                self._next_report += self.cfg.hop_s
            self._next_tick = round(self._next_tick + self.cfg.tick_s, 6)

    # -------------------------------------------------------------------------- blocks
    def _text_block(self, b):
        if self.text is None:
            return
        logits, ms = self.text(b.text)
        self._emit(self.session.observe("text", b.t_end, logits=logits, quality=b.quality, dur=1.0,
                                        meta={"text": b.text, "closed_by": b.reason, "proc_ms": round(ms, 2)}))

    def handle(self, ev: Event):
        if ev.kind == "word":
            for b in self.text_seg.push_word(ev.data["word"], ev.t, ev.data.get("speaker")):
                self._text_block(b)
        elif ev.kind == "line":
            for b in self.text_seg.push_line(ev.data["text"], ev.t, ev.data.get("speaker")):
                self._text_block(b)
        elif ev.kind == "audio" and self.speech is not None:
            for b in self.audio_seg.push(ev.data):
                logits, q, ms = self.speech(b.samples)
                self._emit(self.session.observe("speech", b.t_end, logits=logits, quality=q, dur=b.t_end - b.t_start,
                                                meta={"closed_by": b.reason, "seg": [round(b.t_start, 2), round(b.t_end, 2)],
                                                      "proc_ms": round(ms, 2)}))
        elif ev.kind == "frame" and self.face is not None and self.sampler.accept(ev.t):
            logits, q, ms = self.face(ev.data)
            if logits is not None:                       # no face detected -> modality simply absent at this time
                self._emit(self.session.observe("face", ev.t, logits=logits, quality=q, dur=1.0 / self.sampler.fps,
                                                meta={"proc_ms": round(ms, 2)}))
        elif ev.kind == "block":                         # pre-computed model output (synthetic demo / external models)
            d = ev.data
            self._emit(self.session.observe(d["modality"], ev.t, logits=d.get("logits"), probs=d.get("probs"),
                                            quality=d.get("quality", 1.0), dur=d.get("dur", 1.0), meta=d.get("meta")))

    def run(self, events: Iterable[Event]):
        last = 0.0
        for ev in events:
            self.advance(ev.t)
            self.handle(ev)
            last = ev.t
        for b in self.text_seg.flush():
            self._text_block(b)
        if self.speech is not None:
            for b in self.audio_seg.flush():
                logits, q, ms = self.speech(b.samples)
                self._emit(self.session.observe("speech", b.t_end, logits=logits, quality=q, dur=b.t_end - b.t_start))
        self.advance(last + self.cfg.tick_s)
        return self.records
