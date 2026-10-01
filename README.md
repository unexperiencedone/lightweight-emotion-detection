# Lightweight multimodal emotion detection (edge)

DistilBERT text emotion model (dair-ai/emotion, 6 classes) -> accuracy gate (>= 90 %) -> int8 quantization -> edge simulation,
plus voice-prosody and facial-expression models and a **probabilistic late-fusion** layer that flags ambiguity
(`confident / blend / ambiguous / conflict / uncertain`).

* Full write-up (what, why, how, decision log, limitations): [`docs/TECHNICAL_DOCUMENTATION.md`](docs/TECHNICAL_DOCUMENTATION.md)
* Generated numbers and graphs: [`docs/RESULTS.md`](docs/RESULTS.md)

**Status:** real DistilBERT accuracy is *not yet measured*; the sandbox blocked `huggingface.co`. Everything else is implemented and tested
(see section 1 of the technical documentation). `scripts/run_text.sh` produces the real numbers once the Hub, or local data and weights, are reachable.

```python
from emotion_edge.fusion.fuse import Modality, fuse
f = fuse([Modality("text", text_logits, temperature=T_text),
          Modality("speech", speech_logits, temperature=T_sp, quality=audio_quality),
          None])                                   # a missing modality is fine
print(f.as_dict())   # label, state, per-modality readings, explanation, full posterior
```

Tests: `PYTHONPATH=. pytest -q tests`.
