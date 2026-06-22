from pathlib import Path
from typing import cast
import cv2
import numpy as np
import onnxruntime as ort
import torch


def letterbox(
    image: np.ndarray,
    new_shape: tuple[int, int] = (640, 640),
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """
    Resize and pad an image to a target shape while preserving the aspect ratio.

    :param image: Input BGR image of shape (H, W, 3).
    :param new_shape: Target shape of (H, W).
    :param color: Padding color.
    :return:
        padded: Letterboxed image.
        ratio: Scaling ratio applied to the original image.
        pad: Padding added as (dw, dh) on each side.
    """
    shape = image.shape[: 2]
    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2
    if shape[::-1] != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded, ratio, (dw, dh)


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """
    Convert boxes from center format to corner format.

    :param boxes: Boxes of shape (N, 4), format (cx, cy, w, h).
    :return: Boxes of shape (N, 4), format (top-left x, top-right y, bottom-left x, bottom-right y).
    """
    out = np.empty_like(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return out


class YOLOv5Face:
    """
    YOLOv5 face detector.
    """
    def __init__(
        self,
        model_file: str | None = None,
        session: ort.InferenceSession | None = None,
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.4
    ):
        """
        Initialize YOLOv5 face detector.

        :param model_file: Path tp the exported ONNX model file.
        :param session: Pre-built ONNX runtime session. If None, a new session is created from model_file.
        :param conf_threshold: Confidence threshold for filtering detections.
        :param nms_threshold: NMS IoU threshold.
        """
        self.model_file = model_file
        self.task_name = "detection"
        if session is not None:
            self.session: ort.InferenceSession = session
        else:
            assert self.model_file is not None
            assert Path(self.model_file).exists()
            self.session = ort.InferenceSession(self.model_file, None)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        # YOLOv5-Face raw output layout (16 columns): [cx, cy, w, h, obj, 10 landmarks coordinates, cls]
        self.num_landmarks = 5
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

    def prepare(self, device_id: int, **kwargs):
        """
        Configure the execution provider and runtime thresholds.

        :param device_id: Device ID. Negative value forces CPU execution.
        :param kwargs:
            conf_threshold: Confidence threshold for filtering detections.
            nms_threshold: NMS IoU threshold.
        """
        if device_id < 0:
            self.session.set_providers(["CPUExecutionProvider"])
            conf_threshold = kwargs.get("conf_threshold", None)
            if conf_threshold is not None:
                self.conf_threshold = conf_threshold
            nms_threshold = kwargs.get("nms_threshold", None)
            if nms_threshold is not None:
                self.nms_threshold = nms_threshold

    # noinspection PyMethodMayBeStatic
    def _preprocess(
        self,
        image: np.ndarray,
        input_size: tuple[int, int]
    ) -> tuple[np.ndarray, float, tuple[float, float]]:
        """
        Letterbox, convert BGR -> RGB, normalize, and add a batch dimension.

        :param image: Input BGR image.
        :param input_size: Model input size (W, H).
        :return:
            blob: Preprocessed tensor of shape (1, 3, H, W).
            ratio: Resize ratio used by letterbox.
            pad: Padding (dw, dh) used by the letterbox.
        """
        padded, ratio, pad = letterbox(image, new_shape=(input_size[1], input_size[0]))
        blob = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.ascontiguousarray(np.expand_dims(blob, axis=0))
        return blob, ratio, pad


    def _postprocess(
        self,
        preds: np.ndarray,
        threshold: float,
        ratio: float,
        pad: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Threshold, decode, NMS, and map detections back to original image space.

        :param preds: Raw model output of shape (N, 16).
        :param threshold: Confidence threshold for filtering detections.
        :param ratio: Resize ration from preprocessing.
        :param pad: Padding (dw, dh) from preprocessing.
        :return:
            detections: Array of shape (M, 5), format [x1, y1, x2, y2, score].
            landmarks: Array of shape (M, 5, 2).
        """
        scores = preds[:, 4] * preds[:, 15]         # conf = conf_obj * conf_cls
        keep = scores >= threshold
        preds, scores = preds[keep], scores[keep]
        if preds.shape[0] == 0:
            return np.zeros((0, 5), np.float32), np.zeros((0, self.num_landmarks, 2), np.float32)

        boxes = xywh_to_xyxy(preds[:, :4])
        landmarks = preds[:, 5: 15].reshape(-1, self.num_landmarks, 2)

        # NMS in letterboxed space (cv2 expects [x, y, w, h] top-left + size)
        wh = boxes[:, 2: 4] - boxes[:, 0: 2]
        idxs =  cv2.dnn.NMSBoxes(
            np.c_[boxes[:, :2], wh].tolist(),
            scores.tolist(),
            threshold,
            self.nms_threshold
        )
        idxs = np.array(idxs).reshape(-1)
        if idxs.size == 0:
            return np.zeros((0, 5), np.float32), np.zeros((0, self.num_landmarks, 2), np.float32)
        boxes, scores, landmarks = boxes[idxs], scores[idxs], landmarks[idxs]

        # undo letterbox: subtract pas, divide by ratio
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad[0]) / ratio
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad[1]) / ratio
        landmarks[..., 0] = (landmarks[..., 0] - pad[0]) / ratio
        landmarks[..., 1] = (landmarks[..., 1] - pad[1]) / ratio

        detections = np.hstack([boxes, scores[:, None]]).astype(np.float32)
        return detections, landmarks

    def detect(
        self,
        image: np.ndarray,
        input_size: tuple[int, int] = (640, 640)
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Detect faces and return bounding boxes and landmarks.

        :param image: Input BGR image.
        :param input_size: Model input size (W, H).
        :return:
            bboxes: Bounding boxes of shape (N, 5), format [x1, y1, x2, y2, score].
            landmarks: Landmarks of shape (N, 5, 2) or None when no faces are found.
        """
        blob, ratio, pad = self._preprocess(image, input_size)
        outputs = cast(list[np.ndarray], self.session.run(self.output_names, {self.input_name: blob}))
        detections, landmarks = self._postprocess(outputs[0][0], threshold=self.conf_threshold, ratio=ratio, pad=pad)
        if detections.shape[0] == 0:
            return np.zeros((0, 5), np.float32), None
        return np.int32(detections), np.int32(landmarks)

    def detect_tracking(
        self,
        image: np.ndarray,
        input_size: tuple[int, int] = (128, 128)
    ) -> tuple[torch.Tensor, dict, np.ndarray, np.ndarray |  None]:
        """
        Detect faces and return outputs formatted for the ByteTrack tracker.

        :param image: Input BGR image.
        :param input_size: Model input size (W, H).
        :return:
            outputs: Detections tensor of shape (N, 5), for tracker input.
            image_info: Image metadata with keys "id", "height", "width", "raw_img".
            bbxoes: Bounding boxes of shape (N, 5), format [x1, y1, x2, y2, score].
            landmarks: Landmarks of shape (N, 5, 2) or None when no faces are found.
        """
        height, width = image.shape[:2]
        image_info = {"id": 0, "height": height, "width": width, "raw_img": image}

        blob, ratio, pad = self._preprocess(image, input_size)
        outputs = cast(list[np.ndarray], self.session.run(self.output_names, {self.input_name: blob}))
        detections, landmarks = self._postprocess(outputs[0][0], threshold=self.conf_threshold, ratio=ratio, pad=pad)
        if detections.shape[0] == 0:
            return torch.zeros((0, 5)), image_info, np.zeros((0, 5), np.float32), None
        return torch.tensor(detections), image_info, np.int32(detections), np.int32(landmarks)
