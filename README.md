# Lightweight multimodal emotion detection (edge)

On-device emotion recognition from **text, voice prosody and facial expression**. The system has five layers:

1. **Compact int8 models**, one per modality. The text model is DistilBERT fine-tuned on `dair-ai/emotion`, quantized only after it passes a 90 % accuracy gate.
2. **Block segmentation**, which decides when a sentence, voiced segment or frame is complete and ready for analysis.
3. **A temporal layer**, which gives an **instant emotion** (sticky HMM filter) and an **overall mood over a time window** (Dirichlet evidence, with the probability that the mood label is right), plus a **behavioural pattern** (stable / shifting / volatile / escalating / de-escalating / flat / incongruent).
4. **Probabilistic late fusion**, which weights each modality by its calibrated confidence and input quality and flags ambiguity as `confident`, `blend`, `ambiguous`, `conflict` or `uncertain`.
5. **Edge simulation**: core pinning, contention, memory cap and queueing load tests.

![Live timeline](docs/figures/live_timeline.png)

*Synthetic demo session: raw readings per modality (dots) become a stable fused instant emotion (bottom strip). Masked anger (text says joy, face and voice say anger) is flagged as `conflict` (red squares).*

```mermaid
flowchart LR
  S["words · audio · frames"] --> B["block segmentation"] --> M["int8 models<br/>text · voice · face"]
  M --> F["sticky HMM filter<br/>per modality"] --> I["instant emotion<br/>+ ambiguity state"]
  M --> D["Dirichlet mood<br/>per modality"] --> W["window mood + P(dominant)<br/>+ behavioural pattern"]
```

## Status

| | |
|---|---|
| Real DistilBERT accuracy on dair-ai/emotion | **Not measured yet.** The build sandbox blocked `huggingface.co`. Run `make real-text` or `notebooks/colab_text_run.ipynb` |
| Edge size and latency (full DistilBERT architecture) | Measured: 268 MB → **56.5 MB**; 40.8 → **15.2 ms** per 32-token input on one core (random weights; timing and size are weight-independent) |
| Voice and face models | Code complete and tested on synthetic signals; need RAVDESS / CREMA-D and FER2013 to train |
| Fusion and temporal layer | Implemented, unit-tested, evaluated on simulated data: instant accuracy 0.76 → **0.96**, label flicker 43 → **1.9** switches/min, masking detection 0.23 → **0.70** (versus static fusion) |
| Tests | 27 passing (`make test`) |

## Quickstart

```bash
pip install -r requirements.txt
make test          # 27 tests, no network needed
make live-demo     # end-to-end live pipeline on a scripted session: prints readings, writes JSONL, figures, evaluation
make smoke         # text training pipeline on synthetic data (code path only, NOT a result)
make arch edge latency fusion report     # edge numbers, fusion study, docs/RESULTS.md
make real-text     # real DistilBERT run (needs huggingface.co, or DATA_DIR + MODEL for offline use)
```

Live on your own recordings or devices (see [docs/LIVE_PIPELINE.md](docs/LIVE_PIPELINE.md)):

```bash
python scripts/live.py files --video s.mp4 --audio s.wav --transcript s.jsonl --text-onnx ... --face-onnx ... --speech-onnx ...
python scripts/live.py realtime --seconds 120 --camera 0 ...
```

Library use:

```python
from emotion_edge.live.pipeline import LivePipeline
p = LivePipeline(text=my_text_model, speech=my_speech_model, face=my_face_model, sink=print)
p.run(events)          # records: "block", "instant" (every 0.5 s), "window" (every 5 s)
```

## Documentation

| Document | What's in it |
|---|---|
| [docs/TECHNICAL_DOCUMENTATION.md](docs/TECHNICAL_DOCUMENTATION.md) | Full what / why / how: architecture, text training and the 90 % gate, compression variants, edge simulation, voice and face tracks, fusion, a 23-item decision log, limitations, glossary, references |
| [docs/TEMPORAL_DESIGN.md](docs/TEMPORAL_DESIGN.md) | Block segmentation rules, text strategy decision, sticky HMM filter, Dirichlet mood with P(dominant), behaviour metrics (affect-dynamics literature), parameter rationale, evaluation |
| [docs/LIVE_PIPELINE.md](docs/LIVE_PIPELINE.md) | Running demo / files / real time, input formats, output record schemas, one-core compute budget |
| [docs/RESULTS.md](docs/RESULTS.md) | All measured numbers and graphs, regenerated from result files |

## Layout

| Path | Purpose |
|---|---|
| `emotion_edge/text` | Data loading, DistilBERT training, vocabulary pruning, int8 embeddings, gated pipeline CLI |
| `emotion_edge/speech`, `emotion_edge/vision` | Prosody features + MLP; Haar face detection + mini-Xception CNN; int8 ONNX export |
| `emotion_edge/fusion` | Calibrated late fusion + ambiguity typing; synthetic validation and threshold tuning |
| `emotion_edge/temporal` | Segmenters, sticky filter, Dirichlet mood, behaviour metrics, session, evaluation, figures |
| `emotion_edge/live` | Sources (files, webcam, mic), ONNX predictors, the live pipeline, the demo scenario |
| `emotion_edge/edge` | Edge scenarios and queueing load test |
| `scripts/` | `live.py`, `run_text.sh`, benchmarks, fusion study, `run_speech_face.md` (dataset instructions) |
| `tests/`, `.github/workflows/ci.yml` | Unit, round-trip and integration tests, run on every push |

Datasets are not bundled: dair-ai/emotion (Hugging Face), RAVDESS or CREMA-D (speech), FER2013 (face).

**Responsible use:** outputs describe *expressed* affect probabilistically, not inner states. Run on device, with consent.
