# Lightweight multimodal emotion detection for edge devices: technical documentation

This document records **what** was built, **why** each choice was made, and **how** to reproduce it. It also states, in section 1, what was verified in this repository and what is still pending. Measured numbers live in [`RESULTS.md`](RESULTS.md), which is generated from the result files. Nothing in this document is a measured result unless it says so.

---

## 1. Status: verified versus pending

| Item | Status | Evidence |
|---|---|---|
| Text pipeline code (train, evaluate, accuracy gate, vocabulary pruning, ONNX export, int8, deploy choice) | Works end to end on a tiny random model with synthetic data | `python -m emotion_edge.text.pipeline --smoke` |
| **DistilBERT accuracy on dair-ai/emotion (target >= 90 %)** | **NOT MEASURED** | `huggingface.co` is blocked by the sandbox network policy, so neither the dataset nor the `distilbert-base-uncased` weights could be fetched |
| Quantized-model accuracy drop on real data | NOT MEASURED | depends on the row above |
| Size, memory and latency of the full-size DistilBERT architecture (random weights), fp32 vs int8 vs pruned+int8 | Measured | `results/edge_text_arch.json`; valid because speed and size do not depend on weight values |
| Edge scenarios (core pinning, contention, memory cap, queueing load) | Implemented and run | `emotion_edge/edge/bench.py` |
| Speech-prosody track (features, MLP, int8, actor-independent split) | Code complete and tested on synthetic signals with known pitch and tempo; **not trained on real speech** | `tests/test_speech_vision.py` |
| Face track (Haar detect, mini-Xception CNN, static int8) | Code complete and tested on synthetic images; **not trained on real faces** | `tests/test_speech_vision.py` |
| Probabilistic late fusion with ambiguity typing | Implemented, unit-tested, and validated on **simulated** modality outputs only | `tests/test_fusion.py`, `results/fusion_sim.json` |

The one-command real run is `scripts/run_text.sh`. It needs `huggingface.co` allowed in the environment's network settings, or a local copy of the data and weights (see section 10).

---

## 2. Goals and constraints

1. Fine-tune **DistilBERT** on the six-class `dair-ai/emotion` dataset (sadness, joy, love, anger, fear, surprise) and optimise rounded test accuracy.
2. **Gate:** quantize only if rounded test accuracy is at least 90 %. Verify the quantized model still meets the accuracy budget.
3. Evaluate parameters, size, memory and latency in **simulated edge environments**, with graphs.
4. Make the text model one pluggable **modality** next to voice prosody and facial expression, with the same train, quantize and evaluate steps for each.
5. Combine modalities **probabilistically, not by early fusion**, so that confidence drives the result and **ambiguity can be flagged** for complex or mixed emotions.

---

## 3. Text track

### 3.1 Data (what and why)

`dair-ai/emotion` (config `split`) has 16,000 training, 2,000 validation and 2,000 test tweet-length texts with 6 labels. Properties that drive the design:

- **Class imbalance.** Per the dataset card, joy and sadness dominate and surprise is about 3.6 %. For this reason macro-F1 and a confusion matrix are reported next to accuracy; accuracy alone hides weak classes.
- **Distant-supervision labels.** The labels came from hashtag patterns (Saravia et al., 2018), so some label noise is built in. Published DistilBERT fine-tunes report roughly 92-93 % accuracy, which suggests a ceiling near there. This is context from the literature, not something measured here. The 90 % gate is therefore realistic but leaves little headroom, so quantization loss matters.
- **Short inputs.** Texts are short, so `max_len = 64` is used. The pipeline measures and records the truncation rate (`data.truncation_rate_train` in `results.json`) instead of assuming it is zero.

### 3.2 Model and training (how)

`distilbert-base-uncased` (6 layers, 768 hidden, 66.96 M parameters, of which 23.84 M are word embeddings) with the standard classification head. Training is a plain PyTorch loop (`emotion_edge/text/model.py`).

| Decision | Choice | Reason |
|---|---|---|
| Optimiser | AdamW, weight decay 0.01 (none on bias and LayerNorm), 6 % linear warm-up then linear decay, gradient clip 1.0 | Standard transformer fine-tuning recipe; stable at lr 3e-5 to 5e-5 |
| Batching | Shuffled mega-batches sorted by length inside, then shuffled batches | Dynamic padding cuts CPU cost several-fold without hurting randomness |
| Model selection | Best **validation** accuracy epoch | Test is touched once per run, for reporting only |
| Sweep | lr {3e-5, 5e-5} x epochs {3, 4}, chosen on validation (`scripts/run_text.sh`) | Cheap and enough for this task; test never influences the choice |
| Optional levers for accuracy | Label smoothing (`--label-smoothing`), knowledge distillation from a larger fine-tuned teacher (`--teacher`, KL at T=2, alpha=0.5), multi-seed runs | Off by default. Use them only if the plain recipe lands under the gate |
| Calibration | Temperature scaling fitted on validation logits | Fusion needs probabilities that mean something (section 6) |

"Optimise for rounded-off accuracy" is implemented as follows. The gate compares `round(100 * test_accuracy, 1)` with the threshold, and every table reports accuracy rounded to 0.1 pt next to macro-F1.

### 3.3 Quantization and the accuracy gate

The flow is `train -> evaluate fp32 -> gate (>= 90.0 %) -> variants -> evaluate each -> pick`. If the gate fails, the pipeline writes the reason and **does not quantize**.

Four deployable variants are built and each is evaluated on the same test set:

| Variant | What | Why it exists |
|---|---|---|
| `onnx_fp32` | ONNX export, no compression | Baseline; checks that export itself is lossless |
| `onnx_int8` | ONNX Runtime **dynamic** int8 on MatMul and Gemm (weights int8, activations quantized at run time) | Needs no calibration set; the standard CPU choice for transformers. Static int8 is less safe for attention softmax and LayerNorm outliers |
| `onnx_pruned_fp32` | Embedding table cut to the wordpieces seen in train plus validation; unseen test tokens map to `[UNK]` | The embedding table is 36 % of all parameters. Using train plus validation only keeps test leakage out |
| `onnx_pruned_int8` and `onnx_pruned_int8_emb8` | Pruned vocabulary plus int8 encoder; the second also stores the embedding table as per-row int8 | ORT's dynamic quantizer leaves `Gather` tables in fp32, so a custom `Int8Embedding` (Gather int8, cast, multiply by row scale) is used |

**Selection rule:** the smallest variant whose accuracy drop versus torch fp32 is at most 1.0 pt (`--max-quant-drop`). Why 1.0 pt: against a ceiling near 93 %, a larger drop would erase the benefit of tuning in section 3.2. The report also prints the prediction agreement rate between each variant and the torch model.

### 3.4 Size and parameters (measured, architecture-only)

These come from `scripts/arch_benchmark.py` with random weights. File sizes are exact for the architecture; the weights only influence accuracy.

| Variant | Parameters | File size |
|---|---|---|
| fp32 | 66.96 M | 267.7 MB |
| dynamic int8 | 66.96 M | 138.6 MB |
| vocab pruned to a hypothetical 15 k tokens, fp32 | about 55 M | 220.0 MB |
| pruned 15 k + int8 encoder + int8 embeddings | about 55 M | **56.5 MB (4.7x smaller than fp32)** |

The real kept-vocabulary size comes from the training data and is recorded in `results.json` under `vocab_pruning`. 15 k is an assumption used only for this size and latency measurement.

---

## 4. Edge-device simulation

### 4.1 What is simulated (and what is not)

The sandbox has no ARM board. `emotion_edge/edge/bench.py` therefore **constrains a host CPU so it behaves like a weaker device**. Each scenario runs in a fresh subprocess so settings cannot leak between runs.

| Scenario | Mechanism | Models |
|---|---|---|
| `host_4core` | 4 cores, 4 threads | A desktop-class upper bound |
| `sbc_2core` | `sched_setaffinity` to 2 cores, 2 intra-op threads | Small single-board computer or phone little cluster |
| `sbc_1core` | 1 core, 1 thread | Worst-case cheap SBC; also what a real-time audio or video pipeline leaves for NLP |
| `sbc_1core_contended` | 1 core plus a busy-loop process on the same core | Noisy neighbour or thermal throttling |
| `mem_capped_1core` | `RLIMIT_AS` of 1.1 GB | Memory-limited device. See the caveat below |

On top of the scenarios, a **queueing load test** (Poisson arrivals into one worker, service times from the measurements) gives p50, p95 and p99 latency including waiting time, plus the arrival rate at which the system saturates.

**Limits, stated plainly.**
- This does not emulate the ISA (ARM NEON versus x86 AVX), cache hierarchy or clock. Absolute milliseconds are *host* numbers. Relative comparisons (fp32 versus int8, 1 core versus 2) transfer better than absolute ones.
- `--slowdown` accepts a host-to-device ratio you measure once on real hardware. Projected values are then labelled as projections.
- `RLIMIT_AS` limits address space, not resident memory, so the cap in `mem_capped_1core` is **not binding** for any model here. The evidence for memory fit is the reported **peak RSS** (fp32 about 0.43 GB, int8 0.28 GB, pruned int8 about 0.16 GB). Treat the fp32 model as marginal for a 512 MB device and the pruned int8 model as comfortable.
- Four-thread numbers on very short sequences are noisy and sometimes slower than two threads, because threading overhead exceeds the work. Use the 1-core and 2-core rows for decisions.

### 4.2 Findings so far (architecture-only, details and graphs in RESULTS.md)

- **int8 is about 2.7x faster than fp32 on one core** at 32 tokens (p50 40.8 ms to 15.2 ms; 66.7 ms to 25.8 ms at 64 tokens), and about 2x smaller on disk (3x with the encoder-only figure, 4.7x with pruning and int8 embeddings).
- **Load test (1 core, service time taken at 64 tokens, Poisson arrivals):** fp32 is stable up to 10 Hz (p95 345 ms) and saturates before 20 Hz; int8 holds p95 at 90 ms at 20 Hz and saturates before 40 Hz. The arrival rate for a single text stream is far below this, but several concurrent streams on one core are not.
- **Vocabulary pruning and embedding quantization do not change compute** (latency is about the same as plain int8). They cut file size and resident memory: peak RSS 279 MB to 157 MB. They matter for flash and RAM-limited devices, not for speed.
- **Contention doubles latency** on a shared core, so budgets must include a safety factor.
- **The neural networks are not the bottleneck in the multimodal pipeline.** Measured on one core: audio prosody feature extraction (3 s clip) about 37 ms, Haar face detection (320x240) about 45 ms, versus under 0.5 ms for the speech and face networks and about 0.2 ms for fusion. Optimising the feature extractors and the detector (lower frame rate, smaller input, ROI tracking between frames) pays off more than shrinking the networks further.

---

## 5. Voice-prosody and facial modalities (same steps as text)

Same recipe for each: **train, calibrate (temperature), export ONNX, quantize, evaluate (accuracy, macro-F1, size, latency), save logits for fusion.**

### 5.1 Voice prosody (`emotion_edge/speech`)

- **Why prosody features and not a speech encoder:** wav2vec2 and HuBERT have 95 M or more parameters, which would swamp the budget set by a 67 M-parameter text model. Affect in the voice is largely paralinguistic (pitch level and movement, loudness, tempo, voicing, spectral tilt), and about 120 summary statistics capture it.
- **Features** (`features.py`): F0 in semitones relative to 100 Hz (speaker-agnostic scale), its deltas, log-RMS and deltas, voiced ratio, jitter and shimmer proxies, onset rate (syllable-rate proxy), pause ratio, spectral centroid, rolloff, zero-crossing rate, and 13 MFCC means and standard deviations. Each statistic is summarised by mean, std, min, max, range, p10, p90 and slope. F0 uses YIN, which is much faster than pYIN; the summary statistics do not need pYIN's voicing posteriors.
- **Verified:** a synthetic tone one octave higher reads 12 +/- 1.5 semitones higher, and higher tempo gives a higher onset rate (unit tests).
- **Model:** 29 k-parameter MLP with input standardisation baked into the graph, feature-noise augmentation and label smoothing. Dynamic int8 export.
- **Evaluation protocol (important):** train, validation and test are split **by actor**, never by utterance. Utterance-level splits leak speaker identity and inflate accuracy by a large margin. Datasets: RAVDESS (8 classes) or CREMA-D (6 classes).
- **Quality signal for fusion:** `audio_quality()` (SNR estimate multiplied by voiced fraction) scales the modality's reliability so that noisy or silent audio is discounted.

### 5.2 Facial expression (`emotion_edge/vision`)

- **Detection:** OpenCV Haar cascade, largest face, crop to 48x48 grayscale. It is cheap and CPU-only. A tiny face lowers the reliability score fed to fusion; no face means the modality is dropped. A learned detector would be more robust on non-frontal faces, at higher cost.
- **Model:** mini-Xception-style CNN with depthwise-separable blocks and global average pooling, 66 k parameters, 0.10 MB after static int8 (QDQ, per-channel, calibrated on 200 training images). Convolutions need static quantization; dynamic quantization only covers MatMul and Gemm.
- **Training:** FER2013 (`fer2013.csv`), PublicTest for model selection and PrivateTest for the reported number. Flip, shift and brightness augmentation, label smoothing, one-cycle learning rate.
- **Expectation management:** FER2013 labels are noisy, and small from-scratch CNNs are usually reported in the mid-60s to low-70s percent range. This is a literature expectation, not a measurement here.

---

## 6. Probabilistic late fusion (`emotion_edge/fusion/fuse.py`)

### 6.1 Why late, why probabilistic

Early fusion (concatenating features before one classifier) needs time-aligned, complete inputs and a jointly trained model. On an edge device the modalities arrive at different rates, drop out (no face in frame, silence, no transcript), and come from different datasets with different label sets. Early fusion also hides disagreement between modalities, and disagreement is the very signal needed for ambiguity detection. Here each modality keeps its own calibrated model, and fusion works on posteriors.

### 6.2 Algorithm

For each available modality *m* with calibrated distribution p_m over its own labels:

1. **Calibrate:** p_m = softmax(logits / T_m), with T_m fitted on validation data.
2. **Project to the shared 8-class space** (anger, disgust, fear, joy, love, neutral, sadness, surprise) with a mapping matrix (`labels.py`). Examples: calm maps to neutral; happy maps to joy 0.85 plus love 0.15.
3. **Coverage handling:** classes a modality cannot express receive the uniform level 1/|covered|. That means "no information", not "impossible". Text has no neutral class, so a text opinion must not veto neutral from face and voice (unit-tested).
4. **Discount:** q~_m = r_m q_m + (1 - r_m)/K, with reliability r_m = input quality x base weight (SNR, face size, token count). A blurry face or noisy audio then loses influence smoothly, and a missing modality is simply absent.
5. **Pool:** log P(y) = log prior(y) + sum over m of log q~_m(y), then normalise. This is a product of experts (log-opinion pool). The output is a full posterior, not just a label.

### 6.3 Ambiguity typing

A confidence threshold only says "unsure". Complex emotions need to know **why** the system is unsure, because the right response differs:

| State | Trigger | Meaning and suggested action |
|---|---|---|
| `confident` | Top-1 posterior >= `conf_p`, margin >= `margin`, experts agree | Act on the label |
| `blend` | Top-2 are close **and** valence/arousal-compatible (for example joy and love) | Report a mixed emotion |
| `ambiguous` | Top-2 are close but incompatible (for example joy versus anger) | Collect more evidence or ask |
| `conflict` | Maximum pairwise Jensen-Shannon divergence between modalities, weighted by reliability, exceeds `conflict_jsd` | Incongruence: sarcasm, masking, a posed smile, or a failing sensor. Surface each modality's own reading |
| `uncertain` | Normalised entropy above `entropy`, or top-1 below 0.30 | Abstain |

Compatibility uses a coarse valence/arousal placement of the eight classes (`labels.VALENCE_AROUSAL`). `conflict` is evaluated first because a peaked pooled posterior can hide two experts that strongly disagree.

### 6.4 Validation on simulated modality outputs

`emotion_edge/fusion/simulate.py` generates modality logits whose strengths **are assumptions**: roughly text 74 %, speech 62 % and face 60 % on the 8-class space. Text tops out near 75 % there because it cannot express two of the eight classes. Scenarios are clean, dropout, blend and incongruent. This validates the **logic**, not real-world accuracy. Results (6,000 held-out samples, thresholds tuned on a separate 1,500-sample set by maximising F1 of "flag for review"):

- Fused accuracy on clean and dropout samples is 0.865, versus 0.741 for the best single modality.
- Accuracy among samples reported as `confident` is 0.961 (coverage 0.50); among flagged samples it is 0.673. The selective-prediction curve rises monotonically as coverage falls.
- Flag rates by scenario: clean 0.37, dropout 0.55, blend 0.64, incongruent 0.80. The flag rate on clean data is high because the simulated speech and face channels are weak, so many samples are legitimately uncertain.

**Threshold caveat.** The tuned thresholds are specific to the simulator. On real data, re-tune them with `tune_thresholds`, using the saved validation logits of the three real models and **human ambiguity annotations** (for example, "this clip is a blend" or "face and voice disagree"). The "should flag" label in the simulator is a stand-in for that.

### 6.5 Real-time use

`fuse()` takes a list of whichever modalities are available, so partial input works. Fusion costs about 0.2 ms. For streams, run it on a sliding window (for example, the last 3 s of audio, the most recent face crop, the most recent utterance) and smooth the posterior over time. Temporal smoothing is not implemented yet.

---

## 7. Decision log

| # | Decision | Alternatives considered | Why |
|---|---|---|---|
| D1 | Plain PyTorch loop instead of HF `Trainer` | `Trainer` | Full control over length bucketing, the best-validation checkpoint and optional distillation, with no hidden callbacks |
| D2 | Select on validation, report test once per run | Pick on test | Avoids optimistic bias; the gate is judged on an unbiased estimate |
| D3 | ONNX Runtime as the deployment format | TFLite, ExecuTorch, torch dynamic quantization | One format for transformer, MLP and CNN; good CPU kernels on x86 and ARM; quantization tooling for both dynamic and static modes |
| D4 | Dynamic int8 for transformer and MLP, static (QDQ) int8 for CNN | Static for everything | Dynamic quantization needs no calibration data and avoids activation-range problems in attention; convolutions need static to benefit |
| D5 | Custom int8 embedding table | Leave it fp32 | The table is 36 % of parameters and ORT leaves it fp32 |
| D6 | Vocabulary pruning from train plus validation tokens | Keep the full vocab | Cuts about 12 M parameters; unseen test tokens become `[UNK]`, and accuracy impact is measured, not assumed |
| D7 | Accept a quantized variant only if the drop is at most 1.0 pt | Accept any | Keeps the gate meaningful after compression |
| D8 | Prosody features and a small MLP instead of a speech foundation model | wav2vec2, HuBERT | Parameter budget and latency; these cues are paralinguistic |
| D9 | Actor-independent splits for speech | Random utterance splits | Prevents speaker leakage |
| D10 | Late fusion by log-opinion pool with a reliability discount | Early fusion, Dempster-Shafer, learned stacking | Handles dropout and mismatched label sets, keeps disagreement observable, needs no joint training data |
| D11 | Shared 8-class canonical space with coverage masks | Intersect labels (6 or fewer classes) | Keeps neutral and disgust, which face and voice can express, without letting text veto them |
| D12 | Typed ambiguity states instead of one threshold | Entropy only | Blend, conflict and uncertainty call for different actions |
| D13 | Report macro-F1, the confusion matrix and calibration (ECE) next to accuracy | Accuracy only | Class imbalance and the need for trustworthy probabilities in fusion |
| D14 | Do not report accuracy numbers from synthetic or random-weight runs as results | Show smoke-test accuracy | They only prove the code path. Everything synthetic is labelled in the JSON (`"synthetic": true`) and in RESULTS.md |

---

## 8. Limitations and risks

- **Text domain shift.** dair-ai texts are tweets with hashtag-derived labels. Transcripts of speech, chat messages and long text differ. Validate on in-domain text before trusting scores in a product.
- **Emotion recognition is inference of expressed affect, not of inner state.** Facial expression and voice vary with culture, individual and context. Outputs should be treated as probabilistic signals, and the `conflict` and `blend` states exist partly for this reason.
- **Fairness and consent.** FER2013 and acted-speech corpora are demographically narrow. Measure per-group error before deployment, and process video and audio on-device with consent. This design keeps all inference local, which helps.
- **Simulator thresholds** are not real-data thresholds (section 6.4).
- **Edge numbers** are host-based proxies (section 4.1).
- **Speech to text** is out of scope. The text modality assumes a transcript exists. An on-device ASR would add its own latency and size budget.
- No temporal modelling yet (section 6.5).

---

## 9. Repository map

```
emotion_edge/labels.py            label sets, canonical space, mapping matrices, valence/arousal
emotion_edge/common/              ONNX export and quantization helpers, metrics, temperature scaling
emotion_edge/text/                data loading, model + vocab pruning + int8 embedding, pipeline CLI
emotion_edge/speech/              prosody features, MLP training and export
emotion_edge/vision/              mini-Xception, face detection and crop, training and export
emotion_edge/fusion/              fuse.py (algorithm), simulate.py (synthetic validation + threshold tuning)
emotion_edge/edge/bench.py        edge scenarios + queueing simulation
emotion_edge/report.py            figures + docs/RESULTS.md
scripts/                          run_text.sh, arch_benchmark.py, multimodal_latency.py, run_fusion_sim.py, run_edge_bench.py
tests/                            fusion rules, prosody features, speech and face export/quantization round trips
```

---

## 10. Reproduce

```bash
python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
export PYTHONPATH=$PWD
pytest -q tests                                   # 13 tests; no network needed
python -m emotion_edge.text.pipeline --smoke --out artifacts/smoke --epochs 6 --lr 1e-3   # code-path check (synthetic)
python scripts/arch_benchmark.py artifacts/arch   # random-weight, full-size DistilBERT: sizes
python scripts/run_edge_bench.py --kind text --out results/edge_text_arch.json \
    fp32=artifacts/arch/fp32.onnx int8=artifacts/arch/int8.onnx pruned15k_int8_emb8=artifacts/arch/pruned15k_int8_emb8.onnx
python scripts/multimodal_latency.py results/multimodal_latency.json
python scripts/run_fusion_sim.py
python -m emotion_edge.report                     # regenerates docs/RESULTS.md and docs/figures/*

# REAL text run (needs huggingface.co, or DATA_DIR + MODEL for offline use)
scripts/run_text.sh
# speech / face: see scripts/run_speech_face.md
```

Run order for the full real result: `run_text.sh`, then the speech and face trainers, then re-tune fusion thresholds on the saved real validation logits, then `python -m emotion_edge.report`.
