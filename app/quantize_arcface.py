import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnxconverter_common import float16
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
sys.path.insert(0, str(ROOT_DIR / "src" / "realtime_face_recognition_ros2" / "realtime_face_recognition_ros2"))

from core.constants import IMAGE_EXTS, BACKBONES
from core.recognizer.arcface import iresnet_inference


def preprocess(face_bgr: np.ndarray, size: tuple[int, int] = (112, 112)) -> np.ndarray:
    """
    Preprocess an aligned face crop into a model input blob.
    MUST match ArcFaceONNX.get_feature exactly:
    (pixel - 127.5) / 127.5, BGR -> RGB, CHW, shape (1, 3, 112, 112).

    :param face_bgr: Aligned face crop in BGR format.
    :param size: Target (W, H) spatial size the model expects.
    :return: Float32 blob of shape (1, 3, H, W).
    """
    return cv2.dnn.blobFromImage(
        face_bgr, 1.0 / 127.5, size, (127.5, 127.5, 127.5), swapRB=True
    )


class FaceCalibrationReader(CalibrationDataReader):
    """
    Feeds aligned face crops through the same preprocessing used at inference.
    """

    def __init__(
        self,
        image_dir: str,
        input_name: str,
        size: tuple[int, int] = (112, 112),
        limit: int | None = None,
    ):
        """
        :param image_dir: Folder of aligned face crops, search recursively.
        :param input_name: ONNX graph input tensor name to feed each blob under.
        :param size: Model input (W, H) used for preprocessing.
        :param limit: Max number of images to use; None use all found.
        """
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


def verify_embeddings(
    ref_path: Path,
    test_path: Path,
    image_dir: str,
    label: str,
    num_samples: int = 20,
    size: tuple[int, int] = (112, 112)
):
    """
    Sanity check: run the same aligned crops through the reference FP32 model and the converted model, and report
    cosine similarity between their L2-normalized embeddings.
    FP16 should stay ~0.999; a healthy INT8 keeps mean >= ~0.97.

    :param ref_path: Reference FP32 ONNX model.
    :param test_path: Converted model (FP16 or INT8) to compare against the reference.
    :param image_dir: Folder of aligned crops to evaluate on.
    :param label: Display label for the converted precision (e.g., "FP16", "INT8").
    :param num_samples: Max number of crop to evaluate.
    :param size: Model input (W, H) used for preprocessing.
    """
    paths = [
        p for p in sorted(Path(image_dir).rglob("*"))
        if p.suffix.lower() in IMAGE_EXTS
    ][: num_samples]
    if not paths:
        print("No images for verification; skipping.")
        return
    ref = ort.InferenceSession(str(ref_path), providers=["CPUExecutionProvider"])
    test = ort.InferenceSession(str(test_path), providers=["CPUExecutionProvider"])
    ref_input = ref.get_inputs()[0].name
    test_input = test.get_inputs()[0].name

    similarity_scores = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        blob = preprocess(img, size)
        a = np.asarray(ref.run(None, {ref_input: blob})[0])[0].astype(np.float32)
        b = np.asarray(test.run(None, {test_input: blob})[0])[0].astype(np.float32)
        a /= np.linalg.norm(a)
        b /= np.linalg.norm(b)
        similarity_scores.append(float(np.dot(a, b)))

    if len(similarity_scores) == 0:
        print("No readable images for verification; skipping.")
        return

    similarity_scores = np.array(similarity_scores)
    print(
        f"FP32 vs {label} cosine similarity over {len(similarity_scores)} crops: "
        f"mean={similarity_scores.mean():.4f} min={similarity_scores.min():.4f} max={similarity_scores.max():.4f}"
    )

    if similarity_scores.mean() < 0.97:
        print(
            f"WARNING: mean similarity < 0.97 - {label} may have degraded accuracy. "
            f"For INT8, try CalibrationMethod.Percentile/Entropy or more representative "
            f"calibration crop, or fall back to FP32/FP16."
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
):
    """
    Load the IResNet .pth checkpoint and export an FP32 ONNX with a dynamic batch
    dimension (so the quantized model can later be batched).

    :param pth_path: Source PyTorch .pth checkpoint.
    :param backbone: IResNet backbone variant matching the checkpoint (e.g. "r100").
    :param onnx_path: Destination path for the exported FP32 ONNX.
    :param image_size: Square input resolution (H = W).
    :param device: Torch device to build/export on (e.g. "cpu").
    :param opset: ONNX opset version for the export.
    :param input_name: Name to assign the ONNX graph input.
    :param output_name: Name to assign the ONNX graph output.
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


def export_fp16_onnx(fp32_path: Path, fp16_path: Path):
    """
    Convert an FP32 ONNX to FP16.
    keep_io_types=True leaves the graph input/output as float32, so the FP16 model stays a drop-in replacement for
    ArcFaceONNX (which feeds a float32 blob) - only the internal weights/compute are halved.

    :param fp32_path:
    :param fp16_path:
    """
    model = onnx.load(str(fp32_path))
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model_fp16, str(fp16_path))


def export_int8_onnx(
    fp32_path: Path,
    int8_path: Path,
    calib_dir: str,
    target: str,
    limit: int,
    data_before: set[Path]
):
    """
    Shape-infer, calibrate over aligned crops, and statically quantize to INT8.
    Clean up the intermediate prepped model and stray external-dat sidecars.

    :param fp32_path: Source FP32 ONNX model to quantize.
    :param int8_path: Destination path for the INT8 ONNX.
    :param calib_dir: Folder of aligned 112x112 crops used for calibration.
    :param target: Optimization target: "cpu" (U8S8 + QOperator) or "trt" (symmetric INT8 + QDQ for TensorRT).
    :param limit: Max number of calibration images to use.
    :param data_before: Snapshot of *.data files present before quantization.
        used to identify and delete only newly created sidecars.
    """
    prepped = fp32_path.with_name(fp32_path.stem.replace("_fp32", "") + "_prep.onnx")

    # 1) shape inference + graph cleanup (required before static quantization).
    #    Symbolic shape inference can choke on some graphs; fall back to skipping it.
    try:
        quant_pre_process(str(fp32_path), str(prepped))
    except Exception as exc:
        print(f"quant_pre_process: symbolic shape inference failed ({exc}); "
              f"retrying with skip_symbolic_shape=True")
        quant_pre_process(str(fp32_path), str(prepped), skip_symbolic_shape=True)

    # 2) discover the real input tensor name from the prepped model
    sess = ort.InferenceSession(str(prepped), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    # 3) calibration reader over your aligned crops
    reader = FaceCalibrationReader(calib_dir, input_name, limit=limit)

    # 5) static INT8 quantization. Target picks the format/activation type:
    #    cpu  -> U8S8 (uint8 act, int8 weight) + QOperator: stays on the x86
    #            ONNX Runtime fast integer-GEMM path.
    #    trt  -> symmetric int8 act/weight + QDQ: required by TensorRT on Jetson.
    if target == "cpu":
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
        model_output=str(int8_path),
        calibration_data_reader=reader,
        quant_format=quant_format,
        activation_type=activation_type,
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
        calibrate_method=CalibrationMethod.MinMax,
        extra_options=extra
    )

    # cleanup the prepped intermediate (+ its external-data sidebar)
    prepped.unlink(missing_ok=True)
    Path(str(prepped) + ".data").unlink(missing_ok=True)

    cwd = Path.cwd()
    (cwd / "sym_shape_infer_temp.onnx").unlink(missing_ok=True)
    output_data = Path(str(int8_path) + ".data")
    for stray in cwd.glob("*.data"):
        if stray not in data_before and stray.resolve() != output_data.resolve():
            stray.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert an ArcFace IResNet .pth checkpoint to an FP32, FP16, or INT8 ONNX model.")
    p.add_argument("-i", "--input", required=True, help="PyTorch checkpoint, e.g. weights/recognition/arcface_r100.pth")
    p.add_argument("-b", "--backbone", choices=BACKBONES, default="r100",help="IResNet backbone variant matching the checkpoint (default: r100).")
    p.add_argument("-c", "--calib-dir", required=True, help="Folder of ALIGNED 112x112 face crops for calibration.")
    p.add_argument(
        "-p", "--precision", choices=("fp32", "fp16", "int8"), default="int8",
        help="Output precision. fp32: plain export. fp16: half-precision weights "
             "(float32 I/O). int8: static quantization (needs --calib-dir). (default: int8)",
    )
    p.add_argument("-o", "--output", default=None, help="Output .onnx (default: <input stem>_<precision>.onnx).")
    p.add_argument("--image-size", type=int, default=112, help="Square input resolution H=W (default: 112).")
    p.add_argument("--opset", type=int, default=17, help="ONNX opset version for the intermediate FP32 export (default: 17).")
    p.add_argument("--input-name", default="input", help="ONNX graph input name (default: input).")
    p.add_argument("--output-name", default="embedding", help="ONNX graph output name (default: embedding).")
    p.add_argument("--limit", type=int, default=500, help="Max calibration images to use (default: 500).")
    p.add_argument(
        "--target", choices=("cpu", "trt"), default="cpu",
        help="Optimization target. 'cpu': U8S8 + QOperator for the x86 ONNX Runtime "
             "CPU EP (fastest on a laptop). 'trt': symmetric INT8 + QDQ for "
             "TensorRT on Jetson. (default: cpu)",
    )
    p.add_argument("--keep-fp32", action="store_true", help="Keep the intermediate FP32 ONNX (named <input stem>_fp32.onnx) instead of deleting it.")
    p.add_argument("--check-samples", type=int, default=20, help="Crops to use for the FP32-vs-INT8 cosine sanity check (default: 20).")
    p.add_argument("--no-check", dest="check", action="store_false", default=True, help="Skip the FP32-vs-INT8 cosine similarity sanity check.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pth_path = Path(args.input)
    if not pth_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {pth_path}")
    if args.precision == "int8" and not args.calib_dir:
        raise ValueError("--calib-dir is required for --precision int8 (needed for calibration).")

    output_model = (
        Path(args.output) if args.output
        else pth_path.with_name(pth_path.stem + "_int8.onnx")
    )

    # for fp32 the export is the final output; otherwise it is an intermediate
    fp32_model = output_model if args.precision == "fp32" else pth_path.with_name(pth_path.stem + "_fp32.onnx")

    # snapshot stray external-data files so we can clean up the ones the
    # quantization pipeline dumps into the cwd
    cwd = Path.cwd()
    data_before = set(cwd.glob("*.data"))

    # 1) .pth -> FP32 ONNX (dynamic batch), always needed as the base
    export_fp32_onnx(
        pth_path, args.backbone, fp32_model, args.image_size,
        device="cpu", opset=args.opset,
        input_name=args.input_name, output_name=args.output_name,
    )

    if args.precision == "fp32":
        print(f"Exported FP32 model -> {output_model}")
        return

    # 2) convert to the requested precision
    if args.precision == "fp16":
        export_fp16_onnx(fp32_model, output_model)
    else:
        export_int8_onnx(
            fp32_model, output_model, args.calib_dir, args.target, args.limit, data_before
        )

    # 3) optional accuracy check against the FP32 reference (need images)
    if args.check:
        if args.calib_dir:
            verify_embeddings(
                fp32_model, output_model, args.calib_dir, label=args.precision.upper(), num_samples=args.check_samples
            )
        else:
            print("No --calib-dir given; skipping the FP32-vs-output accuracy check.")

    # 4) cleanup intermediates FP32 model unless asked to keep it
    if not args.keep_fp32:
        fp32_model.unlink(missing_ok=True)
        Path(str(fp32_model) + ".data").unlink(missing_ok=True)
    else:
        print(f"Kept FP32 model -> {fp32_model}")
    print(f"Exported {args.precision.upper()} model -> {output_model}")


if __name__ == "__main__":
    main()
