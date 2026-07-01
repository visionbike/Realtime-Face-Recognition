import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnxconverter_common import float16
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

# make core importable when run from the project root
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from core.constants import IMAGE_EXTS


def preprocess(image_bgr: np.ndarray, size: tuple[int, int] = (640, 640)) -> np.ndarray:
    """
    Preprocess a full scene image into a model input blob.
    MUST match ScrfdONNX.get_feature exactly:
    letterbox-pad into a (H, W) canvas, then blobFromImage(scale=1/128, mean=127.5, BGR->RGB).

    :param image_bgr: Input scene image in BGR format.
    :param size: Target (W, H) spatial size the model expects (fixed 640x640 here).
    :return: Float32 blob of shape (1, 3, H, W).
    """
    h, w = image_bgr.shape[:2]
    img_ratio = float(h) / w
    model_ratio = float(size[1]) / size[0]
    if img_ratio > model_ratio:
        new_h = size[1]
        new_w = int(new_h / img_ratio)
    else:
        new_w = size[0]
        new_h = int(new_w * model_ratio)
    resized = cv2.resize(image_bgr, (new_w, new_h))
    padded = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    padded[:new_h, :new_w, :] = resized
    return cv2.dnn.blobFromImage(
        padded, 1.0 / 128, size, (127.5, 127.5, 127.5), swapRB=True
    )


class SceneCalibrationReader(CalibrationDataReader):
    """
    Feeds full scene images through the same preprocessing used at inference.
    """

    def __init__(
        self,
        image_dir: str,
        input_name: str,
        size: tuple[int, int] = (640, 640),
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


def verify_outputs(
    ref_path: Path,
    test_path: Path,
    image_dir: str,
    label: str,
    num_samples: int = 20,
    size: tuple[int, int] = (640, 640)
):
    """
    Sanity check: run the same aligned crops through the reference FP32 model and the converted model, and report
    cosine similarity between their L2-normalized embeddings.
    FP16 should stay ~0.999; a healthy INT8 keeps mean >= ~0.99.
    Detection heads are sensitive, so treat a lower bar for INT8 with caution.

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
    out_names = [o.name for o in ref.get_outputs()]

    scores = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        blob = preprocess(img, size)
        a_outs = ref.run(out_names, {ref_input: blob})
        b_outs = test.run(out_names, {test_input: blob})
        # concat all heads into one vector and compare direction
        a = np.concatenate([np.asarray(o).ravel() for o in a_outs]).astype(np.float32)
        b = np.concatenate([np.asarray(o).ravel() for o in b_outs]).astype(np.float32)
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
        scores.append(float(np.dot(a, b) / denom))

    if len(scores) == 0:
        print("No readable images for verification; skipping.")
        return

    similarity_scores = np.array(scores)
    print(
        f"FP32 vs {label} cosine similarity over {len(similarity_scores)} crops: "
        f"mean={similarity_scores.mean():.4f} min={similarity_scores.min():.4f} max={similarity_scores.max():.4f}"
    )

    if similarity_scores.mean() < 0.99:
        print(
            f"WARNING: mean similarity < 0.99 - {label} may have degraded accuracy. "
            f"For INT8, try CalibrationMethod.Percentile/Entropy or more representative "
            f"calibration crop, or fall back to FP32/FP16."
        )

def export_fp16_onnx(fp32_path: Path, fp16_path: Path):
    """
    Convert an FP32 ONNX to FP16.
    keep_io_types=True leaves the graph input/output as float32, so the FP16 model stays a drop-in replacement for
    ArcFaceONNX (which feeds a float32 blob) - only the internal weights/compute are halved.

    :param fp32_path: Source FP32 ONNX model.
    :param fp16_path: Destination FP16 model.
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
    exclude_nodes: list[str],
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
    :param exclude_nodes: Node names to leave in float (e.g. sensitive output convs).
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
    reader = SceneCalibrationReader(calib_dir, input_name, limit=limit)

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
        nodes_to_exclude=exclude_nodes or None,
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
    p = argparse.ArgumentParser(description="Convert a SCRFD ONNX detector to an FP16 or INT8 ONNX model.")
    p.add_argument("-i", "--input", required=True,
                   help="Source SCRFD ONNX, e.g. weights/detection/scrfd_2.5g_bnkps.onnx")
    p.add_argument("-c", "--calib-dir", default=None,
                   help="Folder of representative SCENE images for calibration (required for int8).")
    p.add_argument("-p", "--precision", choices=("fp16", "int8"), default="int8",
                   help="Output precision. fp16: half weights (float32 I/O). "
                        "int8: static quantization (needs --calib-dir). (default: int8)")
    p.add_argument("-o", "--output", default=None,
                   help="Output .onnx (default: <input stem>_<precision>.onnx).")
    p.add_argument("--image-size", type=int, default=640,
                   help="Square input resolution H=W (default: 640).")
    p.add_argument("--limit", type=int, default=500,
                   help="Max calibration images to use (default: 500).")
    p.add_argument("--target", choices=("cpu", "trt"), default="trt",
                   help="'cpu': U8S8 + QOperator. 'trt': symmetric INT8 + QDQ for "
                        "TensorRT. (default: trt)")
    p.add_argument("--exclude-nodes", nargs="*", default=[],
                   help="Node names to keep in float (use if INT8 detection accuracy drops).")
    p.add_argument("--check-samples", type=int, default=20,
                   help="Scenes to use for the FP32-vs-output cosine check (default: 20).")
    p.add_argument("--no-check", dest="check", action="store_false", default=True,
                   help="Skip the FP32-vs-output similarity sanity check.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    fp32_model = Path(args.input)
    if not fp32_model.exists():
        raise FileNotFoundError(f"ONNX not found: {fp32_model}")

    if args.precision == "int8" and not args.calib_dir:
        raise ValueError("--calib-dir is required for --precision int8 (needed for calibration).")

    output_model = (
        Path(args.output) if args.output
        else fp32_model.with_name(fp32_model.stem + f"_{args.precision}.onnx")
    )

    cwd = Path.cwd()
    data_before = set(cwd.glob("*.data"))
    size = (args.image_size, args.image_size)

    if args.precision == "fp16":
        export_fp16_onnx(fp32_model, output_model)
    else:
        export_int8_onnx(
            fp32_model, output_model, args.calib_dir, args.target, args.limit, args.exclude_nodes, data_before
        )

    if args.check:
        if args.calib_dir:
            verify_outputs(
                fp32_model, output_model, args.calib_dir, args.precision.upper(), args.check_samples, size
            )
    else:
        print("No --calib-dir given; skipping the FP32-vs-output accuracy check.")

    print(f"Exported {args.precision.upper()} model -> {output_model}")


if __name__ == "__main__":
    main()