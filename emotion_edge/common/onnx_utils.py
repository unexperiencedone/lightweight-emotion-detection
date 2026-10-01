"""Export / quantize / inspect helpers shared by the text, speech and vision models."""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np


def count_params(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    emb = sum(p.numel() for n, p in model.named_parameters() if "embeddings" in n or "embed" in n)
    return {"total": total, "embedding": emb, "non_embedding": total - emb}


def file_mb(path) -> float:
    return os.path.getsize(path) / 1e6


def export_onnx(model, example_inputs: tuple, path, input_names, output_names, dynamic_axes, opset=17):
    import torch
    model.eval()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    kw = dict(input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes,
              opset_version=opset)
    with torch.no_grad():
        try:  # legacy TorchScript exporter: smaller graphs, stable with ORT quantizers
            torch.onnx.export(model, example_inputs, str(path), dynamo=False, **kw)
        except TypeError:
            torch.onnx.export(model, example_inputs, str(path), **kw)
    return path


def _preprocess(src, dst):
    """ORT recommends shape-inference + graph optimisation before quantizing."""
    try:
        from onnxruntime.quantization.shape_inference import quant_pre_process
        quant_pre_process(str(src), str(dst), skip_symbolic_shape=True)
        return dst
    except Exception:
        return src


def quantize_dynamic_int8(src, dst, op_types=("MatMul", "Gemm"), per_channel=False):
    """Weight-only int8, activations quantized on the fly. Best fit for transformers / MLPs on CPU."""
    from onnxruntime.quantization import quantize_dynamic, QuantType
    pre = _preprocess(src, str(src) + ".pre.onnx")
    quantize_dynamic(str(pre), str(dst), weight_type=QuantType.QInt8,
                     op_types_to_quantize=list(op_types), per_channel=per_channel)
    if str(pre) != str(src) and os.path.exists(pre):
        os.remove(pre)
    return dst


def quantize_static_int8(src, dst, calib_batches, input_name: str):
    """Full int8 (weights + activations) with calibration data. Needed for conv nets."""
    from onnxruntime.quantization import (quantize_static, CalibrationDataReader, QuantType,
                                          QuantFormat)

    class _Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(calib_batches)

        def get_next(self):
            b = next(self.it, None)
            return None if b is None else {input_name: b.astype(np.float32)}

    pre = _preprocess(src, str(src) + ".pre.onnx")
    quantize_static(str(pre), str(dst), _Reader(), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8, per_channel=True)
    if str(pre) != str(src) and os.path.exists(pre):
        os.remove(pre)
    return dst


def ort_session(path, threads=1):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
