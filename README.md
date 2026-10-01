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

## Quickstart
```bash
pip install -r requirements.txt
make test            # 13 tests, no network needed
make smoke           # text pipeline code path on synthetic data (NOT a result)
make arch edge latency fusion report   # architecture-only edge numbers, fusion simulation, docs/RESULTS.md
make real-text       # real DistilBERT run: needs huggingface.co (or DATA_DIR + MODEL for offline)
```
GPU users: open `notebooks/colab_text_run.ipynb`.

## Layout
| Path | Purpose |
|---|---|
| `emotion_edge/text` | data loading, DistilBERT training, vocab pruning, int8 embeddings, pipeline CLI with accuracy gate |
| `emotion_edge/speech`, `emotion_edge/vision` | prosody features and MLP; face detection and CNN; both export to int8 ONNX |
| `emotion_edge/fusion` | `fuse.py` (calibrated late fusion + ambiguity states), `simulate.py` (synthetic validation, threshold tuning) |
| `emotion_edge/edge` | edge scenarios (core pinning, contention, memory cap) and queueing load test |
| `scripts/` | `run_text.sh`, benchmarks, fusion simulation, `run_speech_face.md` (dataset instructions) |
| `tests/`, `.github/workflows/ci.yml` | unit and round-trip tests, run on every push |
| `docs/` | technical documentation, generated results, figures |

Datasets are not bundled: dair-ai/emotion (Hugging Face), RAVDESS or CREMA-D (speech), FER2013 (face).
