"""Edge-environment simulation for any ONNX model in this repo.

What is simulated (honest scope): host CPU is constrained to look like a weaker device by
  * pinning the process to N cores and N intra-op threads         (core-count limited SBC / phone big-core cluster)
  * spawning busy-loop processes on the same cores                 (noisy neighbour / thermal-throttle style contention)
  * capping address space (RLIMIT_AS)                              (RAM-limited device; fails loudly when exceeded)
  * Poisson request arrivals into one worker                       (queueing: p95 under load, saturation point)
It does NOT emulate ISA (ARM NEON vs x86 AVX), cache sizes or clock. Absolute ms therefore are *host* numbers;
`--slowdown` lets you apply a ratio measured once on real hardware to get a projection (labelled as such).

Each scenario runs in a fresh subprocess so affinity / rlimit / thread settings cannot leak between runs.
"""
from __future__ import annotations
import argparse
import json
import multiprocessing as mp
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

SCENARIOS = {  # name: (cores, threads, contention_procs, mem_cap_mb)
    "host_4core": (4, 4, 0, None),
    "sbc_2core": (2, 2, 0, 1500),
    "sbc_1core": (1, 1, 0, 1000),
    "sbc_1core_contended": (1, 1, 1, 1000),
    "mem_capped_1core": (1, 1, 0, 1100),  # RLIMIT_AS 1.1 GB: process must fit in ~0.5 GB RSS + runtime address-space overhead
}


def _burn():
    x = 0
    while True:
        x += 1


def make_feed(sess, kind, seq_len, texts_ids=None):
    feed = {}
    for i in sess.get_inputs():
        shp = [d if isinstance(d, int) else None for d in i.shape]
        if i.name in ("input_ids",):
            feed[i.name] = np.random.randint(5, 1000, (1, seq_len)).astype(np.int64) if texts_ids is None else texts_ids
        elif i.name == "attention_mask":
            feed[i.name] = np.ones((1, seq_len), np.int64)
        else:
            shp = [1 if s is None else s for s in shp]
            feed[i.name] = np.random.randn(*shp).astype(np.float32)
    return feed


def run_child(args):
    """Executed inside the constrained subprocess; prints one JSON line."""
    cores, threads, burn, mem = args.cores, args.threads, args.contention, args.mem_mb
    if cores:
        os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:cores]))
    hogs = [mp.Process(target=_burn, daemon=True) for _ in range(burn)]
    for h in hogs:
        os.sched_setaffinity(0, os.sched_getaffinity(0))
        h.start()
    from emotion_edge.common.onnx_utils import ort_session
    if mem:
        resource.setrlimit(resource.RLIMIT_AS, (mem * 2 ** 20, mem * 2 ** 20))
    out = {"model": args.model, "cores": cores, "threads": threads, "contention": burn, "mem_cap_mb": mem}
    try:
        sess = ort_session(args.model, threads)
        for L in args.seq_lens:
            feed = make_feed(sess, args.kind, L)
            for _ in range(args.warmup):
                sess.run(None, feed)
            ts = []
            for _ in range(args.iters):
                t = time.perf_counter()
                sess.run(None, feed)
                ts.append((time.perf_counter() - t) * 1000)
            ts = np.array(ts) * args.slowdown
            out[f"len{L}"] = {"mean_ms": ts.mean(), "p50_ms": np.percentile(ts, 50), "p95_ms": np.percentile(ts, 95),
                              "p99_ms": np.percentile(ts, 99), "std_ms": ts.std(), "min_ms": ts.min(), "n": len(ts)}
        out["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        out["service_ms_for_queue"] = float(ts.mean())
        out["status"] = "ok"
    except MemoryError as e:
        out["status"] = "OOM"
    except Exception as e:  # ORT raises its own error types on allocation failure
        out["status"] = "OOM" if "alloc" in str(e).lower() or "memory" in str(e).lower() else f"error: {e}"
    for h in hogs:
        h.terminate()
    print("RESULT" + json.dumps(out, default=float))


def queue_sim(service_ms: np.ndarray, rates_hz, n=4000, seed=0):
    """Single-server FIFO queue with Poisson arrivals and *empirical* service times (M/G/1 by simulation)."""
    rng = np.random.default_rng(seed)
    res = {}
    for lam in rates_hz:
        arr = np.cumsum(rng.exponential(1000.0 / lam, n))
        svc = rng.choice(service_ms, n)
        free, lat = 0.0, np.empty(n)
        for i in range(n):
            start = max(arr[i], free)
            free = start + svc[i]
            lat[i] = free - arr[i]
        res[str(lam)] = {"utilisation": float(min(1.0, lam * svc.mean() / 1000)), "p50_ms": float(np.percentile(lat, 50)),
                         "p95_ms": float(np.percentile(lat, 95)), "p99_ms": float(np.percentile(lat, 99)),
                         "stable": bool(lam * svc.mean() / 1000 < 1.0)}
    return res


def run_all(models: dict[str, str], kind: str, out: str, seq_lens=(16, 32, 64), iters=200, slowdown=1.0,
            scenarios=None, rates=(2, 5, 10, 20, 40)):
    results = {}
    for mname, mpath in models.items():
        results[mname] = {}
        for sname in (scenarios or SCENARIOS):
            cores, th, burn, mem = SCENARIOS[sname]
            cmd = [sys.executable, "-m", "emotion_edge.edge.bench", "--child", "--model", str(mpath), "--kind", kind,
                   "--cores", str(cores), "--threads", str(th), "--contention", str(burn),
                   "--iters", str(iters), "--slowdown", str(slowdown), "--seq-lens", *map(str, seq_lens)]
            if mem:
                cmd += ["--mem-mb", str(mem)]
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            line = [l for l in p.stdout.splitlines() if l.startswith("RESULT")]
            r = json.loads(line[0][6:]) if line else {"status": f"crash: {p.stderr[-300:]}"}
            if r.get("status") == "ok":
                r["queue"] = queue_sim(np.full(500, r["service_ms_for_queue"]) * np.random.default_rng(0).lognormal(0, 0.05, 500), rates)
            results[mname][sname] = r
            print(mname, sname, r.get("status"), {k: round(v["p50_ms"], 2) for k, v in r.items() if k.startswith("len")}, flush=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({"slowdown_factor": slowdown, "kind": kind, "results": results}, indent=2, default=float))
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--kind", default="text")
    ap.add_argument("--cores", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--contention", type=int, default=0)
    ap.add_argument("--mem-mb", type=int, default=0)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--slowdown", type=float, default=1.0)
    ap.add_argument("--seq-lens", type=int, nargs="+", default=[16, 32, 64])
    a = ap.parse_args()
    if a.child:
        run_child(a)
    else:
        raise SystemExit("use run_all() or scripts/run_edge_bench.py")
