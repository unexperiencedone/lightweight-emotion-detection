"""python scripts/run_edge_bench.py --kind text --out results/edge_text.json name=path.onnx [name=path.onnx ...]"""
import argparse
from emotion_edge.edge.bench import run_all

ap = argparse.ArgumentParser()
ap.add_argument("models", nargs="+", help="name=path.onnx")
ap.add_argument("--kind", default="text")
ap.add_argument("--out", required=True)
ap.add_argument("--iters", type=int, default=200)
ap.add_argument("--slowdown", type=float, default=1.0, help="host->device ratio measured on real hardware (projection)")
ap.add_argument("--seq-lens", type=int, nargs="+", default=[16, 32, 64])
a = ap.parse_args()
run_all(dict(m.split("=", 1) for m in a.models), a.kind, a.out, seq_lens=a.seq_lens, iters=a.iters, slowdown=a.slowdown)
