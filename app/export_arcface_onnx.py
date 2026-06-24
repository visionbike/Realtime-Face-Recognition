import sys
from pathlib import Path
import torch

# make core importable when this file is run as app/export_arcface_onnx.py
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src" / "realtime_face_recognition_ros2" / "realtime_face_recognition_ros2"))

from core.recognizer.arcface import iresnet_inference

model = iresnet_inference("r50", "weights/recognition/arcface_r100.pth", "cpu")
dummy = torch.randn(1, 3, 112, 112)

torch.onnx.export(
    model,
    (dummy,),
    "weights/recognition/arcface_r100_int8.onnx",
    dynamo=True,
    external_data=False,        # keep a single inline .onnx file
    input_names=["x"], output_names=["embedding"],
    dynamic_shapes={"x": {0: "batch"}},
    opset_version=17,
)
