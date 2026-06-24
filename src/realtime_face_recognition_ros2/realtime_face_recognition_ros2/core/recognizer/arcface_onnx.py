from pathlib import Path
from typing import cast
import cv2
import numpy as np
import onnxruntime as ort


class ArcFaceONNX:
    """
    ArcFace face recognizer backed by ONNX Runtime.
    """

    def __init__(
        self,
        model_file: str | None = None,
        session: ort.InferenceSession | None = None,
        input_size: tuple[int, int] = (112, 112)
    ):
        """
        :param model_file: Path to the ONNX model file.
        :param session: Pre-built ONNX runtime session. If None, one is created from model_file.
        :param input_size: Model input size (W, H); overridden if the model declares a static shape.
        """
        self.model_file = model_file
        if session is not None:
            self.session = session
        else:
            assert model_file is not None
            assert Path(model_file).exists()
            self.session = ort.InferenceSession(model_file, None)

        inputs = self.session.get_inputs()
        self.input_name = inputs[0].name
        in_shape = inputs[0].shape

        # use the model's static spatial dims when available (shape is [N, 3, H, W])
        if isinstance(in_shape[2], int) and isinstance(in_shape[3], int):
            self.input_size = (in_shape[3], in_shape[2])
        else:
            self.input_size = input_size
        self.output_names = [o.name for o in self.session.get_outputs()]

    def prepare(self, device_id: int):
        """
        Configure the detector execution provider and runtime parameters.

        :param device_id: Device id. Negative value forces CPU execution.
        """
        if device_id < 0:
            self.session.set_providers(["CPUExecutionProvider"])
        else:
            self.session.set_providers(
                [
                    ("TensorrtExecutionProvider", {
                        "trt_int8_enable": False,
                        "trt_fp16_enable": True,  # FP16 fallback for non-QDQ layers
                        "trt_engine_cache_enable": True,
                        "trt_engine_cache_path": "weights/detection/trt_cache",
                    }),
                    ("CUDAExecutionProvider", {"device_id": device_id}),
                    "CPUExecutionProvider",
                ]
            )

    def get_feature(self, face_image: np.ndarray) -> np.ndarray:
        """
        Extract an L2-normalized embedding from an aligned BGR face crop.

        :param face_image: Aligned face crop in BGR format (112x112).
        :return: 1D L2-normalized embedding vector.
        """
        # equivalent to the torch preprocessing (ToTensor + Normalize(0.5, 0.5)):
        #   (pixel / 127.5) - 1, with BGR->RGB. blobFromImage = scale * (img - mean).
        blob = cv2.dnn.blobFromImage(
            face_image, 1.0 / 127.5, self.input_size, (127.5, 127.5, 127.5), swapRB=True
        )
        embedding = cast(np.ndarray, self.session.run(self.output_names, {self.input_name: blob})[0])[0]
        return embedding / np.linalg.norm(embedding)
