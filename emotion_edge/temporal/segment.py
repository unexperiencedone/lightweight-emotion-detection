"""Block segmentation: *when do we stop collecting and send a block for analysis?*

Every modality is a stream; the models need a bounded unit. A good block is
  * long enough to carry affect (one utterance, ~1-6 s of voiced speech, one face crop),
  * short enough not to average over an emotional change,
  * closed by a *natural boundary* (sentence end, pause, speaker change) when one exists, with a hard cap otherwise.

TextSegmenter   words (ASR with timestamps) or typed lines -> utterance blocks
AudioSegmenter  PCM chunks -> voiced segments (energy VAD with adaptive noise floor + hangover)
FrameSampler    video frames -> frames at a fixed analysis rate (detection is the expensive step)
All segmenters are push-based (feed data as it arrives, receive closed blocks) and have flush() for end of stream.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

_FINAL = (".", "!", "?", "…")


@dataclass
class TextBlock:
    t_start: float
    t_end: float
    text: str
    n_words: int
    reason: str            # why the block was closed (for the log / documentation)
    speaker: str | None = None

    @property
    def quality(self) -> float:
        """Short blocks ("ok", "yeah") carry little affect; reliability ramps up to 1 at 6 words."""
        return float(min(1.0, self.n_words / 6.0))


@dataclass
class TextSegmenter:
    pause_s: float = 1.2         # silence/idle gap that ends an utterance
    min_words: int = 3           # below this a sentence end does NOT close the block (merge forward)
    max_words: int = 40          # hard cap (<64 model tokens with wordpiece expansion)
    orphan_gap_s: float = 3.0    # a short block is emitted on its own only if nothing follows for this long
    _words: list = field(default_factory=list)
    _t0: float | None = None
    _t_last: float | None = None
    _speaker: str | None = None

    def _close(self, reason):
        if not self._words:
            return None
        b = TextBlock(self._t0, self._t_last, " ".join(self._words), len(self._words), reason, self._speaker)
        self._words, self._t0 = [], None
        return b

    def push_word(self, word: str, t: float, speaker: str | None = None) -> list[TextBlock]:
        out = []
        if self._words:
            gap = t - self._t_last
            if speaker is not None and speaker != self._speaker:
                out.append(self._close("speaker_change"))
            elif gap >= self.pause_s and (len(self._words) >= self.min_words or gap >= self.orphan_gap_s):
                out.append(self._close(f"pause {gap:.1f}s"))
        if not self._words:
            self._t0, self._speaker = t, speaker
        self._words.append(word)
        self._t_last = t
        if len(self._words) >= self.max_words:
            out.append(self._close("max_words"))
        elif word.endswith(_FINAL) and len(self._words) >= self.min_words:
            out.append(self._close("sentence_end"))
        return [b for b in out if b]

    def push_line(self, line: str, t: float, speaker: str | None = None) -> list[TextBlock]:
        """Typed chat: a line is a candidate block; words share the line timestamp."""
        out = []
        for w in line.split():
            out += self.push_word(w, t, speaker)
        # a typed line ends with an implicit "send"; close it if it is long enough
        if len(self._words) >= self.min_words:
            out.append(self._close("line_end"))
        return [b for b in out if b]

    def tick(self, now: float) -> list[TextBlock]:
        """Call periodically: closes a pending block once the speaker has been quiet long enough."""
        if self._words and now - self._t_last >= max(self.pause_s, self.orphan_gap_s if len(self._words) < self.min_words else 0):
            return [self._close("idle")]
        return []

    def flush(self):
        b = self._close("flush")
        return [b] if b else []


@dataclass
class AudioBlock:
    t_start: float
    t_end: float
    samples: np.ndarray
    reason: str


@dataclass
class AudioSegmenter:
    """Energy VAD on 20 ms frames.

    speech frame  : log-energy > noise_floor + margin_db
    noise floor   : slow-tracking 10th percentile of recent frame energies (adapts to the room)
    hangover      : `min_pause_s` of non-speech is needed before a segment ends (prevents cutting at plosives)
    closure rules : pause >= min_pause_s, or duration >= max_s (force close, carry `overlap_s` into the next one)
    rejection     : segments with < min_s of audio are discarded (too short for prosody statistics)
    """
    sr: int = 16000
    frame_s: float = 0.02
    margin_db: float = 9.0
    min_pause_s: float = 0.40
    min_s: float = 0.8
    max_s: float = 6.0
    overlap_s: float = 0.5
    _buf: list = field(default_factory=list)
    _hist: list = field(default_factory=list)
    _seg: list = field(default_factory=list)
    _seg_t0: float | None = None
    _silence: float = 0.0
    _t: float = 0.0
    _rem: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))

    def _floor(self):
        h = self._hist[-500:]            # last ~10 s
        return np.percentile(h, 10) if len(h) >= 10 else -60.0

    def push(self, pcm: np.ndarray) -> list[AudioBlock]:
        out = []
        x = np.concatenate([self._rem, pcm.astype(np.float32)])
        n = int(self.sr * self.frame_s)
        k = len(x) // n
        self._rem = x[k * n:]
        for i in range(k):
            fr = x[i * n:(i + 1) * n]
            e = 10 * np.log10(np.mean(fr ** 2) + 1e-10)
            self._hist.append(e)
            # no decisions until the noise floor has been measured (first 200 ms) -> no false onset on room noise
            speech = len(self._hist) > 10 and e > self._floor() + self.margin_db and e > -50
            if speech:
                if self._seg_t0 is None:
                    self._seg_t0 = self._t
                self._seg.append(fr)
                self._silence = 0.0
            elif self._seg_t0 is not None:
                self._seg.append(fr)               # keep short pauses inside the segment (they are prosody)
                self._silence += self.frame_s
                if self._silence >= self.min_pause_s:
                    out += self._emit("pause", trim=self._silence)
            self._t += self.frame_s
            if self._seg_t0 is not None and len(self._seg) * self.frame_s >= self.max_s:
                out += self._emit("max_len", carry=self.overlap_s)
        return out

    def _emit(self, reason, trim=0.0, carry=0.0):
        seg = np.concatenate(self._seg) if self._seg else np.zeros(0, np.float32)
        if trim:
            seg = seg[:max(0, len(seg) - int(trim * self.sr))]
        t0 = self._seg_t0
        dur = len(seg) / self.sr
        keep = int(carry / self.frame_s)
        self._seg = self._seg[-keep:] if keep else []
        self._seg_t0 = (self._t - carry) if keep else None
        self._silence = 0.0
        if dur < self.min_s:
            return []
        return [AudioBlock(t0, t0 + dur, seg, reason)]

    def flush(self):
        return self._emit("flush") if self._seg_t0 is not None else []


@dataclass
class FrameSampler:
    """Down-sample a camera/video stream to the analysis rate. Face detection (~20-30 ms/frame on 1 core) dominates the
    face cost, so 4 fps (~8-12 % of one core) is the default; expressions last 0.5-4 s, so 4 fps still sees each one."""
    fps: float = 4.0
    _next: float = 0.0

    def accept(self, t: float) -> bool:
        if t + 1e-9 >= self._next:
            # advance on a fixed grid (no drift); resync if the stream skipped more than one period
            self._next += 1.0 / self.fps
            if self._next <= t:
                self._next = t + 1.0 / self.fps
            return True
        return False
