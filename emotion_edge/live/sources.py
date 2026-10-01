"""Timestamped input events. Offline files are replayed in timestamp order (deterministic, testable);
webcam/microphone sources produce the same events in real time."""
from __future__ import annotations
import heapq
import json
import time
from dataclasses import dataclass, field
from typing import Any, Iterator
import numpy as np


@dataclass(order=True)
class Event:
    t: float
    kind: str = field(compare=False)        # word | line | audio | frame | block
    data: Any = field(compare=False, default=None)


def transcript_events(path) -> Iterator[Event]:
    """JSONL. Either word-level ASR output {"t": 3.2, "word": "really", "speaker": "A"}
    or line-level chat {"t": 10.0, "text": "i am so done with this"}."""
    for line in open(path):
        if line.strip():
            d = json.loads(line)
            yield Event(float(d["t"]), "word" if "word" in d else "line", d)


def wav_events(path, chunk_s=0.1, sr=16000) -> Iterator[Event]:
    import librosa
    y, _ = librosa.load(path, sr=sr, mono=True)
    n = int(chunk_s * sr)
    for i in range(0, len(y), n):
        yield Event((i + n) / sr, "audio", y[i:i + n])          # timestamp = when the chunk is complete


def video_events(path) -> Iterator[Event]:
    import cv2
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield Event(i / fps, "frame", frame)
        i += 1
    cap.release()


def merge(*streams) -> Iterator[Event]:
    return heapq.merge(*streams)


def realtime_events(camera: int | None = 0, mic: bool = True, duration_s: float = 60.0, sr=16000) -> Iterator[Event]:
    """Live capture. Camera via OpenCV; microphone via the optional `sounddevice` package."""
    import queue
    import threading
    q: "queue.Queue[Event]" = queue.Queue()
    t0 = time.monotonic()
    stop = threading.Event()
    if mic:
        import sounddevice as sd
        def cb(indata, frames, ti, status):
            q.put(Event(time.monotonic() - t0, "audio", indata[:, 0].copy()))
        stream = sd.InputStream(samplerate=sr, channels=1, blocksize=int(0.1 * sr), callback=cb)
        stream.start()
    if camera is not None:
        import cv2
        cap = cv2.VideoCapture(camera)
        def grab():
            while not stop.is_set():
                ok, fr = cap.read()
                if ok:
                    q.put(Event(time.monotonic() - t0, "frame", fr))
        threading.Thread(target=grab, daemon=True).start()
    try:
        while time.monotonic() - t0 < duration_s:
            try:
                yield q.get(timeout=0.1)
            except queue.Empty:
                continue
    finally:
        stop.set()
        if mic:
            stream.stop()
        if camera is not None:
            cap.release()
