import torch

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
