import json, numpy as np
from dataclasses import asdict
from emotion_edge.fusion.simulate import make_dataset, evaluate, tune_thresholds
from emotion_edge.fusion.fuse import Thresholds, risk_coverage

val, test = make_dataset(1500, seed=11), make_dataset(6000, seed=1)
th, f1 = tune_thresholds(val)
res = {"synthetic": True, "tuned_thresholds": asdict(th), "val_flag_f1": f1,
       "default_thresholds": asdict(Thresholds())}
for name, t in (("default", Thresholds()), ("tuned", th)):
    r = evaluate(test, t)
    arr = r.pop("_arrays")
    if name == "tuned":
        c, a = risk_coverage(arr["correct"], arr["score"])
        res["risk_coverage"] = {"coverage": c.tolist(), "accuracy": a.tolist()}
        res["overall_acc_no_abstain"] = float(arr["correct"].mean())
    res[name] = r
json.dump(res, open("results/fusion_sim.json", "w"), indent=2, default=float)
r = res["tuned"]
print("unimodal(clean):", {k: round(v, 3) for k, v in r["unimodal_acc_clean"].items()}, "fused(clean):", round(r["fused_acc_clean"], 3))
print("acc|confident", round(r["acc_when_confident"], 3), "acc|flagged", round(r["acc_when_flagged"], 3), "coverage", round(r["coverage_confident"], 3))
print("flag rate by kind:", {k: round(v, 3) for k, v in r["flag_rate_by_kind"].items()})
