import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

# make core importable when run from the project root
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from core.recognizer.arcface import iresnet_inference

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BACKBONES = ("r18", "r34", "r50", "r100")


def preprocess(face_bgr: np.ndarray, size: tuple[int, int] = (112, 112)) -> np.ndarray:
    """
    MUST match ArcFaceONNX.get_feature exactly:
    (pixel - 127.5) / 127.5, BGR -> RGB, CHW, shape (1, 3, 112, 112).
    """
    return cv2.dnn.blobFromImage(
        face_bgr, 1.0 / 127.5, size, (127.5, 127.5, 127.5), swapRB=True
    )


class FaceCalibrationReader(CalibrationDataReader):
    """Feeds aligned face crops through the same preprocessing used at inference."""

    def __init__(
        self,
        image_dir: str,
        input_name: str,
        size: tuple[int, int] = (112, 112),
        limit: int | None = None,
    ):
        self.input_name = input_name
        self.size = size
        paths = [
            p for p in sorted(Path(image_dir).rglob("*"))
            if p.suffix.lower() in IMAGE_EXTS
        ]
        if limit:
            paths = paths[:limit]
        if not paths:
            raise ValueError(f"No calibration images found under {image_dir!r}")
        self.paths = paths
        self._iter = None

    def _generator(self):
        for p in self.paths:
            img = cv2.imread(str(p))
            if img is None:
                continue
            yield {self.input_name: preprocess(img, self.size)}

    def get_next(self):
        if self._iter is None:
            self._iter = self._generator()
        return next(self._iter, None)

    def rewind(self):
        self._iter = None


def verify_int8(
    fp32_path: Path,
    int8_path: Path,
    image_dir: str,
    num_samples: int = 20,
    size: tuple[int, int] = (112, 112),
) -> None:
    """
    Sanity check: run the same aligned crops through the FP32 and INT8 models and
    report cosine similarity between their L2-normalized embeddings. A healthy
    quantization keeps mean similarity high (>= ~0.97).
    """
    paths = [
        p for p in sorted(Path(image_dir).rglob("*"))
        if p.suffix.lower() in IMAGE_EXTS
    ][:num_samples]
    if not paths:
        print("No images for verification; skipping.")
        return

    fp32 = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    int8 = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    fp32_in = fp32.get_inputs()[0].name
    int8_in = int8.get_inputs()[0].name

    sims = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        blob = preprocess(img, size)
        a = fp32.run(None, {fp32_in: blob})[0][0]
        b = int8.run(None, {int8_in: blob})[0][0]
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)
        sims.append(float(np.dot(a, b)))

    if not sims:
        print("No readable images for verification; skipping.")
        return

    sims = np.array(sims)
    print(
        f"FP32 vs INT8 cosine similarity over {len(sims)} crops: "
        f"mean={sims.mean():.4f} min={sims.min():.4f} max={sims.max():.4f}"
    )
    if sims.mean() < 0.97:
        print(
            "WARNING: mean similarity < 0.97 - INT8 may have degraded accuracy. "
            "Try CalibrationMethod.Percentile/Entropy or more representative "
            "calibration crops, or fall back to FP32 for recognition."
        )


def export_fp32_onnx(
    pth_path: Path,
    backbone: str,
    onnx_path: Path,
    image_size: int,
    device: str,
    opset: int,
    input_name: str,
    output_name: str,
) -> None:
    """
    Load the IResNet .pth checkpoint and export an FP32 ONNX with a dynamic batch
    dimension (so the quantized model can later be batched).
    """
    model = iresnet_inference(backbone, str(pth_path), device)
    dummy = torch.randn(1, 3, image_size, image_size, device=device)
    export_kwargs = dict(
        input_names=[input_name],
        output_names=[output_name],
        dynamic_axes={input_name: {0: "batch"}, output_name: {0: "batch"}},
        opset_version=opset,
    )
    # Force the stable TorchScript exporter. The default dynamo exporter routes
    # through onnxscript's version converter, which fails to down-convert dynamic
    # axes to opset 17 and produces a graph that breaks symbolic shape inference.
    try:
        torch.onnx.export(model, (dummy,), str(onnx_path), dynamo=False, **export_kwargs)
    except TypeError:
        # older torch without the `dynamo` kwarg already uses the TorchScript path
        torch.onnx.export(model, (dummy,), str(onnx_path), **export_kwargs)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert an ArcFace IResNet .pth checkpoint to an INT8 ONNX model."
    )
    p.add_argument(
        "-i", "--input", required=True,
        help="PyTorch checkpoint, e.g. weights/recognition/arcface_r100.pth",
    )
    p.add_argument(
        "-b", "--backbone", choices=BACKBONES, default="r100",
        help="IResNet backbone variant matching the checkpoint (default: r100).",
    )
    p.add_argument(
        "-c", "--calib-dir", required=True,
        help="Folder of ALIGNED 112x112 face crops for calibration.",
    )
    p.add_argument(
        "-o", "--output", default=None,
        help="Output INT8 .onnx (default: <input stem>_int8.onnx).",
    )
    p.add_argument(
        "--image-size", type=int, default=112,
        help="Square input resolution H=W (default: 112).",
    )
    p.add_argument(
        "--opset", type=int, default=17,
        help="ONNX opset version for the intermediate FP32 export (default: 17).",
    )
    p.add_argument(
        "--input-name", default="input", help="ONNX graph input name (default: input).",
    )
    p.add_argument(
        "--output-name", default="embedding", help="ONNX graph output name (default: embedding).",
    )
    p.add_argument(
        "--limit", type=int, default=500,
        help="Max calibration images to use (default: 500).",
    )
    p.add_argument(
        "--target", choices=("cpu", "trt"), default="cpu",
        help="Optimization target. 'cpu': U8S8 + QOperator for the x86 ONNX Runtime "
             "CPU EP (fastest on a laptop). 'trt': symmetric INT8 + QDQ for "
             "TensorRT on Jetson. (default: cpu)",
    )
    p.add_argument(
        "--keep-fp32", action="store_true",
        help="Keep the intermediate FP32 ONNX (named <input stem>_fp32.onnx) instead of deleting it.",
    )
    p.add_argument(
        "--check-samples", type=int, default=20,
        help="Crops to use for the FP32-vs-INT8 cosine sanity check (default: 20).",
    )
    p.add_argument(
        "--no-check", dest="check", action="store_false", default=True,
        help="Skip the FP32-vs-INT8 cosine similarity sanity check.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pth_path = Path(args.input)
    if not pth_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {pth_path}")

    output_model = (
        Path(args.output) if args.output
        else pth_path.with_name(pth_path.stem + "_int8.onnx")
    )
    fp32_model = pth_path.with_name(pth_path.stem + "_fp32.onnx")
    prepped = pth_path.with_name(pth_path.stem + "_prep.onnx")

    # 1) .pth -> FP32 ONNX (dynamic batch)
    export_fp32_onnx(
        pth_path, args.backbone, fp32_model, args.image_size,
        device="cpu", opset=args.opset,
        input_name=args.input_name, output_name=args.output_name,
    )

    # snapshot stray external-data files so we can clean up the ones that
    # quant_pre_process / symbolic shape inference dump into the cwd
    cwd = Path.cwd()
    data_before = set(cwd.glob("*.data"))

    # 2) shape inference + graph cleanup (required before static quantization).
    #    Symbolic shape inference can choke on some graphs; fall back to skipping it.
    try:
        quant_pre_process(str(fp32_model), str(prepped))
    except Exception as exc:
        print(f"quant_pre_process: symbolic shape inference failed ({exc}); "
              f"retrying with skip_symbolic_shape=True")
        quant_pre_process(str(fp32_model), str(prepped), skip_symbolic_shape=True)

    # 3) discover the real input tensor name from the prepped model
    sess = ort.InferenceSession(str(prepped), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    # 4) calibration reader over your aligned crops
    reader = FaceCalibrationReader(args.calib_dir, input_name, limit=args.limit)

    # 5) static INT8 quantization. Target picks the format/activation type:
    #    cpu  -> U8S8 (uint8 act, int8 weight) + QOperator: stays on the x86
    #            ONNX Runtime fast integer-GEMM path.
    #    trt  -> symmetric int8 act/weight + QDQ: required by TensorRT on Jetson.
    if args.target == "cpu":
        # QOperator + per_channel hits an ORT static-bias requantize bug
        # (broadcast error on the first conv), so CPU uses per-tensor weights.
        quant_format = QuantFormat.QOperator
        activation_type = QuantType.QUInt8
        per_channel = False
        extra = {}
    else:  # trt
        # QDQ handles per-channel weights fine and TensorRT prefers them.
        quant_format = QuantFormat.QDQ
        activation_type = QuantType.QInt8
        per_channel = True
        extra = {"ActivationSymmetric": True, "WeightSymmetric": True}

    quantize_static(
        model_input=str(prepped),
        model_output=str(output_model),
        calibration_data_reader=reader,
        quant_format=quant_format,
        activation_type=activation_type,
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
        calibrate_method=CalibrationMethod.MinMax,
        extra_options=extra,
    )

    # 6) sanity check FP32 vs INT8 embeddings (before the FP32 model is removed)
    if args.check:
        verify_int8(fp32_model, output_model, args.calib_dir, num_samples=args.check_samples)

    # 7) cleanup intermediates, including external-data (.data) sidecars and the
    #    temp files quant_pre_process leaves in the cwd
    def _remove_onnx(path: Path) -> None:
        path.unlink(missing_ok=True)
        Path(str(path) + ".data").unlink(missing_ok=True)  # external-data sidecar

    _remove_onnx(prepped)
    if not args.keep_fp32:
        _remove_onnx(fp32_model)

    (cwd / "sym_shape_infer_temp.onnx").unlink(missing_ok=True)
    output_data = Path(str(output_model) + ".data")
    for stray in cwd.glob("*.data"):
        if stray not in data_before and stray.resolve() != output_data.resolve():
            stray.unlink(missing_ok=True)

    print(f"Wrote INT8 model -> {output_model}")
    if args.keep_fp32:
        print(f"Kept FP32 model -> {fp32_model}")


if __name__ == "__main__":
    main()
