.PHONY: install test smoke arch edge latency fusion live-demo report real-text real-meld real-cremad
PY ?= python
export PYTHONPATH := $(CURDIR)

install:   ; pip install -r requirements.txt
test:      ; $(PY) -m pytest -q
smoke:     ; $(PY) -m emotion_edge.text.pipeline --smoke --out artifacts/smoke --epochs 6 --lr 1e-3   # synthetic, code path only
arch:      ; $(PY) scripts/arch_benchmark.py artifacts/arch
edge:      ; $(PY) scripts/run_edge_bench.py --kind text --out results/edge_text_arch.json fp32=artifacts/arch/fp32.onnx int8=artifacts/arch/int8.onnx pruned15k_int8_emb8=artifacts/arch/pruned15k_int8_emb8.onnx
latency:   ; $(PY) scripts/multimodal_latency.py results/multimodal_latency.json
fusion:    ; $(PY) scripts/run_fusion_sim.py
live-demo: ; $(PY) scripts/live.py demo --out results/live_demo --quiet   # temporal demo + figures + 10-seed eval
report:    ; $(PY) -m emotion_edge.report
real-text: ; scripts/run_text.sh   # needs huggingface.co (or DATA_DIR + MODEL)
CREMAD ?= data/cremad
MELD ?= data/meld
FEATS ?= artifacts/cremad_feats
MODELS ?= artifacts/cremad_models
real-meld: ; $(PY) scripts/real_meld.py --root $(MELD)
real-cremad:
	OMP_NUM_THREADS=1 $(PY) scripts/real_cremad_extract.py --root $(CREMAD) --out $(FEATS)
	$(PY) scripts/real_cremad.py --root $(CREMAD) --feats $(FEATS) --out $(MODELS)
	$(PY) scripts/real_figures.py --root $(CREMAD) --feats $(FEATS) --models $(MODELS)
