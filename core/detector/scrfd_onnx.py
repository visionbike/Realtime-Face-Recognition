from pathlib import Path
from typing import cast
import cv2
import numpy as np
import onnxruntime as ort
import torch


def softmax(logits: np.ndarray) -> np.ndarray:
    """
    Compute row-wise softmax over a 2D array.

    :param logits: 2D input array of shape (N, C).
    :return: Softmax probabilities of shape (N, C).
    """
    assert len(logits.shape) == 2
    row_max = np.max(logits, axis=-1)
    row_max = row_max[:, np.newaxis]    # necessary step to do broadcasting
    exp_shifted = np.exp(logits - row_max)
    row_sum = np.sum(exp_shifted, axis=1)
    row_sum = row_sum[:, np.newaxis]
    return exp_shifted / row_sum


def distances_to_bboxes(points: np.ndarray, distance: np.ndarray, max_shape: tuple | None = None) -> np.ndarray:
    """
    Decode anchor-based distance predictions to bounding boxes.

    :param points: Anchor center points of shape (N, 2), format [x, y].
    :param distance: Predicted distances to box edges of shape (N, 4), format [left, top, right, bottom].
    :param max_shape: Image shape (H, W) used to clamp box coordinates.
    :return: Decoded bounding box coordinates of shape (N, 4), format [x1, y1, x2, y2].
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clip(min=0, max=max_shape[1])
        y1 = y1.clip(min=0, max=max_shape[0])
        x2 = x2.clip(min=0, max=max_shape[1])
        y2 = y2.clip(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)


def distances_to_keypoints(points: np.ndarray, distance: np.ndarray, max_shape: tuple | None = None) -> np.ndarray:
    """
    Decode anchor-based distance predictions to facial keypoints.

    :param points: Anchor center points of shape (N, 2), format [x, y].
    :param distance: Predicted distances to keypoints of shape (N, K*2), where K is the number of keypoints.
    :param max_shape: Image shape (H, W) used to clamp keypoint coordinates.
    :return: Decoded keypoint coordinates of shape (N, K*2).
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        if max_shape is not None:
            px = px.clip(min=0, max=max_shape[1])
            py = py.clip(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


class SCRFDONNX:
    """
    SCRFD face detector
    """
    def __init__(
        self,
        model_file: str | None = None,
        session: ort.InferenceSession | None = None,
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.4
    ):
        """
        Initialize SCRFD face detector.

        :param model_file: Path to the ONNX model file.
        :param session: Pre-built ONNX runtime session. If None, a new session is created from model_file.
        :param conf_threshold: Confidence threshold for filtering detections.
        :param nms_threshold: NMS IoU threshold.
        """
        self.model_file = model_file
        self.task_name = "detection"
        self.batched = False
        if session is not None:
            self.session: ort.InferenceSession = session
        else:
            assert self.model_file is not None
            assert Path(self.model_file).exists()
            self.session = ort.InferenceSession(self.model_file, None)
        self.center_cache = {}
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

        # attributes populated by _init_vars()
        self.input_size = None
        self.input_name = None
        self.output_names = None
        self.use_kpts = False
        self._num_anchors = 1
        self.feat_map_count = None      # number of feature map classification heads per FPN level
        self._feat_stride_fpn = None
        self._init_vars()

    def _init_vars(self):
        """
        Parse model input/output metadata and configure detector parameters.

        Sets input size, output names, FPN strides, anchor counts, and whether the model supports keypoints and batched output.
        """
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        if isinstance(input_shape[2], str):
            self.input_size = None
        else:
            self.input_size = tuple(input_shape[2: 4][::-1])
        input_name = input_cfg.name
        outputs = self.session.get_outputs()
        if len(outputs[0].shape) == 3:
            self.batched = True
        output_names = []
        for output in outputs:
            output_names.append(output.name)
        self.input_name = input_name
        self.output_names = output_names
        self.use_kpts = False
        self._num_anchors = 1
        if len(outputs) == 6:
            self.feat_map_count = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
        elif len(outputs) == 9:
            self.feature_map_count = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
            self.use_kpts = True
        elif len(outputs) == 10:
            self.feature_map_count = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
        elif len(outputs) == 15:
            self.feature_map_count = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
            self.use_kpts = True
        else:
            raise ValueError(f"Unsupported number of model outputs: {len(outputs)}")

    def prepare(self, device_id: int, **kwargs):
        """
        Configure the detector execution provider and runtime parameters.

        :param device_id: Device id. Negative value forces CPU execution.
        :param kwargs:
            nms_threshold (float): NMS IoU threshold.
            input_size (tuple): Input image size (W, H). Ignored if already set by the model.
        """
        if device_id < 0:
            self.session.set_providers(["CPUExecutionProvider"])
        else:
            self.session.set_providers(
                [
                    ("TensorrtExecutionProvider", {
                        "trt_int8_enable": True,
                        "trt_fp16_enable": True,  # FP16 fallback for non-QDQ layers
                        "trt_engine_cache_enable": True,
                        "trt_engine_cache_path": "weights/recognition/trt_cache",
                    }),
                    ("CUDAExecutionProvider", {"device_id": device_id}),
                    "CPUExecutionProvider",
                ]
            )
        conf_threshold = kwargs.get("conf_threshold", None)
        if conf_threshold is not None:
            self.conf_threshold = conf_threshold
        nms_threshold = kwargs.get("nms_threshold", None)
        if nms_threshold is not None:
            self.nms_threshold = nms_threshold
        input_size = kwargs.get("input_size", None)
        if input_size is not None:
            if self.input_size is not None:
                print("warning: det_size is already set in scrfd model, ignore")
            else:
                self.input_size = input_size

    def forward(
        self,
        image: np.ndarray,
        threshold: float
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        """
        Run a single forward pass and decode raw model outputs.

        :param image: Preprocessed input image of shape (H, W, 3).
        :param threshold: Confidence threshold for filtering detections.
        :return:
            scores_list: Confidence scores per FPN level.
            bboxes_list: Decoded bounding boxes per FPN level.
            kpts_list: Decoded keypoints per FPN level (empty if model does not support keypoints).
        """
        scores_list = []
        bboxes_list = []
        kpts_list = []
        input_size = tuple(image.shape[0: 2][::-1])
        blob = cv2.dnn.blobFromImage(image, 1.0 / 128, input_size, (127.5, 127.5, 127.5), swapRB=True)
        net_outs = cast(list[np.ndarray], self.session.run(self.output_names, {self.input_name: blob}))

        input_height = blob.shape[2]
        input_width = blob.shape[3]
        feature_map_count = self.feature_map_count
        for idx, stride in enumerate(self._feat_stride_fpn):
            kpt_preds = None
            # if model support batch dim, take first output
            if self.batched:
                scores = net_outs[idx][0]
                bbox_preds = net_outs[idx + feature_map_count][0]
                bbox_preds = bbox_preds * stride
                if self.use_kpts:
                    kpt_preds = net_outs[idx + feature_map_count * 2][0] * stride
            # if model doesn't support batching take output as is
            else:
                scores = net_outs[idx]
                bbox_preds = net_outs[idx + feature_map_count]
                bbox_preds = bbox_preds * stride
                if self.use_kpts:
                    kpt_preds = net_outs[idx + feature_map_count * 2] * stride

            height = input_height // stride
            width = input_width // stride
            key = (height, width, stride)
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape((-1, 2))
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers

            pos_ids = np.where(scores >= threshold)[0]
            bboxes = distances_to_bboxes(anchor_centers, bbox_preds)
            pos_scores = scores[pos_ids]
            pos_bboxes = bboxes[pos_ids]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            if self.use_kpts:
                assert kpt_preds is not None
                kpts = distances_to_keypoints(anchor_centers, kpt_preds)
                kpts = kpts.reshape((kpts.shape[0], -1, 2))
                pos_kpts = kpts[pos_ids]
                kpts_list.append(pos_kpts)
        return scores_list, bboxes_list, kpts_list

    def nms(self, dets: np.ndarray) -> list[int]:
        """
        Apply non-maximum suppression to remove overlapping detections.

        :param dets: Detections of shape (N, 5), format [x1, y1, x2, y2, score].
        :return: Indices of kept detections.
        """
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            ids = np.where(ovr <= self.nms_threshold)[0]
            order = order[ids + 1]
        return keep

    def detect(
        self,
        image: np.ndarray,
        input_size: tuple[int, int] = (640, 640),
        max_num: int = 0,
        metric: str = "default"
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Detect faces in an input image and return bounding boxes and landmarks.

        :param image: Input BGR image.
        :param input_size: Model input size (W, H).
        :param max_num: Maximum number of detections to return. 0 means no limit.
        :param metric: Face selection metric when max_num is set.
            "max": select by area.
            "default": select by area minus center offset.
        :return:
            bboxes: Bounding boxes of shape (N, 5), format [x1, y1, x2, y2, score].
            landmarks: Facial keypoints of shape (N, 5, 2).
        """
        assert input_size is not None or self.input_size is not None
        input_size = self.input_size if input_size is None else input_size
        assert input_size is not None, "input_size must be set on the model or passed explicitly."

        im_ratio = float(image.shape[0]) / image.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / image.shape[0]
        resized_img = cv2.resize(image, (new_width, new_height))
        padded_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        padded_img[:new_height, :new_width, :] = resized_img

        scores_list, bboxes_list, kpts_list = self.forward(padded_img, self.conf_threshold)

        scores = np.vstack(scores_list)
        order = scores.ravel().argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale
        kpts = None
        if self.use_kpts:
            kpts = np.vstack(kpts_list) / det_scale
        pre_nms_dets = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_nms_dets = pre_nms_dets[order, :]
        keep = self.nms(pre_nms_dets)
        detections = pre_nms_dets[keep, :]
        if self.use_kpts:
            assert kpts is not None
            kpts = kpts[order, :, :]
            kpts = kpts[keep, :, :]
        else:
            kpts = None
        if 0 < max_num < detections.shape[0]:
            area = (detections[:, 2] - detections[:, 0]) * (detections[:, 3] - detections[:, 1])
            img_center = image.shape[0] // 2, image.shape[1] // 2
            offsets = np.vstack(
                [
                    (detections[:, 0] + detections[:, 2]) / 2 - img_center[1],
                    (detections[:, 1] + detections[:, 3]) / 2 - img_center[0],
                ]
            )
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            if metric == "max":
                values = area
            else:
                values = (area - offset_dist_squared * 2.0)  # some extra weight on the centering
            bindex = np.argsort(values)[::-1]  # some extra weight on the centering
            bindex = bindex[0: max_num]
            detections = detections[bindex, :]
            if kpts is not None:
                kpts = kpts[bindex, :]

        bboxes = np.int32(detections)
        landmarks = np.int32(kpts) if kpts is not None else None

        return bboxes, landmarks

    def detect_tracking(
        self,
        image: np.ndarray,
        input_size: tuple[int, int] = (128, 128),
        max_num: int = 0,
        metric: str = "default"
    ) -> tuple[torch.Tensor, dict, np.ndarray, np.ndarray | None]:
        """
        Detect faces and return outputs formatted for the ByteTrack tracker.

        :param image: Input BGR image.
        :param input_size: Model input size (W, H).
        :param max_num: Maximum number of detections to return. 0 means no limit.
        :param metric: Face selection metric when max_num is set.
            "max": select by area.
            "default": select by area minus center offset.
        :return:
            outputs: Raw detections tensor of shape (N, 5), for tracker input.
            image_info: Image metadata with keys "id", "height", "width", "raw_img".
            bboxes: Bounding boxes of shape (N, 5), format [x1, y1, x2, y2, score].
            landmarks: Facial keypoints of shape (N, 5, 2).
        """
        assert input_size is not None or self.input_size is not None
        input_size = self.input_size if input_size is None else input_size
        assert input_size is not None, "input_size must be set on the model or passed explicitly."

        height, width = image.shape[:2]
        image_info = {"id": 0, "height": height, "width": width, "raw_img": image}

        img_ratio = float(image.shape[0]) / image.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]
        if img_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / img_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * img_ratio)
        det_scale = float(new_height) / image.shape[0]
        resized_img = cv2.resize(image, (new_width, new_height))
        padded_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        padded_img[:new_height, :new_width, :] = resized_img

        scores_list, bboxes_list, kpts_list = self.forward(padded_img, self.conf_threshold)

        scores = np.vstack(scores_list)
        order = scores.ravel().argsort()[::-1]
        bboxes = np.vstack(bboxes_list)
        kpts = None
        if self.use_kpts:
            kpts = np.vstack(kpts_list)
        pre_nms_dets = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_nms_dets = pre_nms_dets[order, :]
        keep = self.nms(pre_nms_dets)
        detections = pre_nms_dets[keep, :]
        if self.use_kpts:
            assert kpts is not None
            kpts = kpts[order, :, :]
            kpts = kpts[keep, :, :]
        else:
            kpts = None
        if 0 < max_num < detections.shape[0]:
            area = (detections[:, 2] - detections[:, 0]) * (detections[:, 3] - detections[:, 1])
            img_center = image.shape[0] // 2, image.shape[1] // 2
            offsets = np.vstack(
                [
                    (detections[:, 0] + detections[:, 2]) / 2 - img_center[1],
                    (detections[:, 1] + detections[:, 3]) / 2 - img_center[0],
                ]
            )
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            if metric == "max":
                values = area
            else:
                values = (
                        area - offset_dist_squared * 2.0
                )  # some extra weight on the centering
            bindex = np.argsort(values)[::-1]  # some extra weight on the centering
            bindex = bindex[0:max_num]
            detections = detections[bindex, :]
            if kpts is not None:
                kpts = kpts[bindex, :]

        bboxes = np.int32(detections / det_scale)
        landmarks = np.int32(kpts / det_scale) if kpts is not None else None

        return torch.tensor(detections), image_info, bboxes, landmarks