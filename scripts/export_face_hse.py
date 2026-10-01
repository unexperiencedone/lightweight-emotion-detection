"""Edge export of the AffectNet-pretrained face model: HSEmotion EfficientNet-B0 graph + our trained head folded in as one
Gemm (standardisation and logistic regression merged), output = 7 FACE_LABELS logits. Then static int8 (QDQ) and checks.

    python scripts/export_face_hse.py --root <cremad> --feats <feats>
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from emotion_edge.labels import FACE_LABELS
from emotion_edge.realdata import cremad as C
from emotion_edge.common import onnx_utils as ou
from face_hse import prep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--feats", required=True)
    a = ap.parse_args()
    import onnx
    from onnx import helper, numpy_helper, TensorProto
    F = Path(a.feats)
    m = onnx.load(F / "hse_enet_b0_8_best_afew.onnx")
    head = pickle.load(open(F / "face_hse_head.pkl", "rb"))       # our own file
    sc, clf = head["scaler"], head["clf"]
    W = np.zeros((len(FACE_LABELS), 1280), np.float32); b = np.full(len(FACE_LABELS), -30.0, np.float32)
    coef = clf.coef_ / sc.scale_                                    # fold StandardScaler into the linear layer
    W[clf.classes_] = coef
    b[clf.classes_] = clf.intercept_ - coef @ sc.mean_
    feat = next(n.output[0] for n in m.graph.node if n.op_type == "Flatten")
    m.graph.initializer.extend([numpy_helper.from_array(W, "emo_W"), numpy_helper.from_array(b, "emo_b")])
    m.graph.node.append(helper.make_node("Gemm", [feat, "emo_W", "emo_b"], ["logits"], transB=1))
    del m.graph.output[:]
    m.graph.output.append(helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch_size", len(FACE_LABELS)]))
    # the published graph is opset 11; per-channel QDQ needs DequantizeLinear(axis) from opset 13
    from onnx import version_converter
    if m.opset_import[0].version < 13:
        m = version_converter.convert_version(m, 13)
    onnx.save(m, F / "face_hse_fp32.onnx")
    Fz = np.load(F / "faces112.npz"); crops, clip = Fz["crops"], Fz["clip"]
    df = C.metadata(a.root); df["split"] = C.actor_split(df)
    sp = df.split.to_numpy()[clip]
    rng = np.random.default_rng(0)
    calib = [prep(crops[i:i + 1]) for i in rng.choice(np.where(sp == "train")[0], 200, replace=False)]
    ou.quantize_static_int8(F / "face_hse_fp32.onnx", F / "face_hse_int8.onnx", calib, "input")
    te = rng.choice(np.where(sp == "test")[0], 1500, replace=False)
    yf = C.face_index(df.code)[clip[te]]
    res = {}
    ref = None
    for prec in ("fp32", "int8"):
        s4 = ou.ort_session(F / f"face_hse_{prec}.onnx", 4)
        L = np.concatenate([s4.run(None, {"input": prep(crops[te[i:i + 64]])})[0] for i in range(0, len(te), 64)])
        s1 = ou.ort_session(F / f"face_hse_{prec}.onnx", 1)
        x = prep(crops[te[:1]]); ms = []
        for _ in range(30):
            t = time.perf_counter(); s1.run(None, {"input": x}); ms.append((time.perf_counter() - t) * 1000)
        res[prec] = {"test_frame_acc_1500": float((L.argmax(1) == yf).mean()), "size_mb": ou.file_mb(F / f"face_hse_{prec}.onnx"),
                     "ms_per_frame_1thread_p50": float(np.median(ms[5:]))}
        if ref is None:
            ref = L.argmax(1)
        res[prec]["agreement_with_fp32"] = float((L.argmax(1) == ref).mean())
        print(prec, res[prec], flush=True)
    out = Path("results/upgrade.json"); r = json.loads(out.read_text()); r["face_hse_edge"] = res; out.write_text(json.dumps(r, indent=2, default=float))


if __name__ == "__main__":
    main()
