from typing import Any
import sys
import time
import threading
from pathlib import Path
import cv2
import numpy as np
import yaml

# make core importable when this file is run as app/recognize_onnx.py
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from core.detector.scrfd_onnx import SCRFDONNX
from core.recognizer.arcface_onnx import ArcFaceONNX
from core.recognizer.feature_store import read_features, compare_embeddings
from core.aligner.alignment import align_face
from core.tracker.byte_tracker import BYTETracker
from core.tracker.visualize import plot_tracking

# ONNX Runtime device id: negative -> CPU, >=0 -> CUDA device index
DEVICE_ID = -1

# asset locations
CONFIG_PATH = ROOT_DIR / "cfgs" / "config.yaml"
DETECTOR_WEIGHTS = ROOT_DIR / "weights" / "detection" / "scrfd_2.5g_bnkps.onnx"
RECOGNIZER_WEIGHTS = ROOT_DIR / "weights" / "recognition" / "arcface_r100_int8.onnx"
FEATURES_PATH = ROOT_DIR / "datasets" / "features"

# model input size used for both detection-tracking and the tracker rescale
TRACKING_INPUT_SIZE = (640, 640)


def load_config(config_path: Path) -> dict:
    """
    Load the YAML config; return an empty dict if no config is found.

    :param config_path: Path to the config file.
    :return: Parsed configuration as a dict.
    """
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_models(config: dict) -> tuple[SCRFDONNX, ArcFaceONNX]:
    """
    Instantiate the SCRFD detector and the ONNX ArcFace recognizer.
    :param config: Parsed config dict.
    :return: Initialized face detector and recognizer.
    """
    det_cfg = config.get("detector", {})
    detector = SCRFDONNX(
        model_file=str(DETECTOR_WEIGHTS),
        conf_threshold=det_cfg.get("conf_threshold", 0.5),
        nms_threshold=det_cfg.get("nms_threshold", 0.4)
    )
    detector.prepare(DEVICE_ID)
    recognizer = ArcFaceONNX(model_file=str(RECOGNIZER_WEIGHTS))
    recognizer.prepare(DEVICE_ID)
    return detector, recognizer


class FaceRecognizer:
    """
    Threaded webcam face recognition: one thread detects + tracks, another recognizer,
    tracked faces, and the main thread displays annotated frames.
    """
    def __init__(
        self,
        config: dict,
        detector: SCRFDONNX,
        recognizer: ArcFaceONNX,
        image_names: np.ndarray,
        image_embeddings: np.ndarray,
    ):
        """
        :param config: Parsed config dict.
        :param detector: Initialized SCRFD detector.
        :param recognizer: Initialized ONNX ArcFace recognizer.
        :param image_names: Enrolled names of shape (N,).
        :param image_embeddings: Enrolled embeddings of shape (N, D).
        """
        self.detector = detector
        self.recognizer = recognizer
        self.image_names = image_names
        self.image_embeddings = image_embeddings

        recognizer_config = config.get("recognizer", {})
        self.similarity_threshold = recognizer_config.get("similarity_threshold", 0.5)
        self.match_iou_threshold = recognizer_config.get("match_iou_threshold", 0.9)
        # re-recognize throttle: known tracks are cached forever;
        # UNKNOWN tracks are retried ar most once per interval instead of every loop iteration
        self.unknown_retry_interval = recognizer_config.get("unknown_retry_interval", 1)
        self.unknown_fast_interval = recognizer_config.get("unknown_fast_interval", 0.1)  # fast (just appeared)
        self.unknown_fast_retries = recognizer_config.get("unknown_fast_retries", 5)  # how many fast tries
        self.edge_margin = recognizer_config.get("edge_margin", 8)
        self.min_face_size = recognizer_config.get("min_face_size", 60)

        tracker_config = config.get("tracker", {})
        self.aspect_ratio_threshold = tracker_config.get("aspect_ratio_threshold", 1.6)
        self.min_box_area = tracker_config.get("min_box_area", 10)
        self.tracker = BYTETracker(
            track_threshold=tracker_config.get("track_threshold", 0.5),
            match_threshold=tracker_config.get("match_threshold", 0.8),
            dup_iou_dist_threshold=tracker_config.get("dup_iou_dist_threshold", 0.15),
            frame_rate=30
        )

        # shared state guarded by _lock; threads stop when _stop is set
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._latest_frame: np.ndarray | None = None
        self.id_face_mapping: dict[int, str] = {}
        self._last_attempt: dict[int, float] = {}
        self._attempts: dict[int, float] = {}
        self.data_mapping: dict[str, Any] = {
            "raw_image": None,
            "tracking_image": None,
            "detection_bboxes": [],
            "detection_landmarks": [],
            "tracking_ids": [],
            "tracking_bboxes": []
        }

    @staticmethod
    def _mapping_bbox(box1: np.ndarray, box2: np.ndarray) -> float:
        """
        Compute IoU between two (x_min, y_min, x_max, y_max) boxes.

        :param box1: First bounding box.
        :param box2: Second bounding box.
        :return: IoU score.
        """
        x_min_inter = max(box1[0], box2[0])
        y_min_inter = max(box1[1], box2[1])
        x_max_inter = min(box1[2], box2[2])
        y_max_inter = min(box1[3], box2[3])

        intersection_area = max(0, x_max_inter - x_min_inter + 1) * max(0, y_max_inter - y_min_inter + 1)
        area_box1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        area_box2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
        union_area = area_box1 + area_box2 - intersection_area
        return intersection_area / union_area if union_area > 0 else 0.0

    def _recognize_face(self, face_image: np.ndarray) -> tuple[float, str | None]:
        """
        Recognize an aligned face crop against the feature store.

        :param face_image: Aligned face crop (BGR, 112x112).
        :return: The best similarity score, matched name or None.
        """
        # ArcFaceONNX takes the BGR crop directly (blobFromImage handles BGR -> RGB + scaling)
        query_emb = self.recognizer.get_feature(face_image)
        score, idx = compare_embeddings(query_emb, self.image_embeddings)
        return score, self.image_names[idx]

    def _process_tracking(self, frame: np.ndarray, frame_id: int, fps: float):
        """
        Detect and track faces in one frame, then publish the shared state.

        :param frame: Input BGR frame.
        :param frame_id: Current frame index.
        :param fps: Current frames-per-second estimate.
        """
        outputs, img_info, bboxes, landmarks = self.detector.detect_tracking(
            image=frame, input_size=TRACKING_INPUT_SIZE
        )

        tracking_tlwhs = []
        tracking_ids = []
        tracking_bboxes = []

        if outputs is not None and len(outputs) > 0:
            online_targets = self.tracker.update(
                outputs,
                (img_info["height"], img_info["width"]),
                TRACKING_INPUT_SIZE,
            )
            for target in online_targets:
                tlwh = target.tlwh
                vertical = tlwh[2] / tlwh[3] > self.aspect_ratio_threshold
                too_small = tlwh[2] < self.min_face_size or tlwh[3] < self.min_face_size
                if tlwh[2] * tlwh[3] > self.min_box_area and not vertical or not too_small:
                    tracking_bboxes.append([tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]])
                    tracking_tlwhs.append(tlwh)
                    tracking_ids.append(target.track_id)
        # prune per-track state to the currently active tracks so id_face_mapping
        # and _last_attempt don't accumulate dead track ids over a long session
        with self._lock:
            active = set(tracking_ids)
            self.id_face_mapping = {i: n for i, n in self.id_face_mapping.items() if i in active}
            self._last_attempt = {i: t for i, t in self._last_attempt.items() if i in active}
            self._attempts = {i: a for i, a in self._attempts.items() if i in active}
            # default not-yet-recognized tracks to "..." for display only
            names = {i: self.id_face_mapping.get(i, "...") for i in tracking_ids}

        if tracking_ids:
            tracking_image = plot_tracking(
                img_info["raw_img"], tracking_tlwhs, tracking_ids,
                names=names, frame_id=frame_id + 1, fps=fps,
            )
        else:
            tracking_image = img_info["raw_img"]

        with self._lock:
            self.data_mapping.update(
                raw_image=img_info["raw_img"],
                tracking_image=tracking_image,
                detection_bboxes=bboxes,
                detection_landmarks=landmarks,
                tracking_ids=tracking_ids,
                tracking_bboxes=tracking_bboxes,
            )

    def _capture_loop(self, source):
        """
        Read frames as fast as the camera allows; keep only the freshest one so a
        slow consumer never falls behind on a stale buffered frame.

        :param source: OpenCV VideoCapture source (camera index or path).
        """
        cap = cv2.VideoCapture(source)
        try:
            while not self._stop.is_set():
                ok, img = cap.read()
                if not ok:
                    break
                with self._lock:
                    self._latest_frame = img    # overwrite, drop the previous frame
        finally:
            cap.release()
            self._stop.set()

    def _tracking_loop(self):
        """
        Detect + track on the most recent frame, skipping any that piled up.
        """
        start_time = time.time_ns()
        frame_count = 0
        fps = -1.0
        frame_id = 0

        while not self._stop.is_set():
            with self._lock:
                img = self._latest_frame
                self._latest_frame = None
            if img is None:
                time.sleep(0.001)
                continue

            self._process_tracking(img, frame_id, fps)

            frame_count += 1
            if frame_count >= 30:
                fps = 1e9 * frame_count / (time.time_ns() - start_time)
                frame_count = 0
                start_time = time.time_ns()
                # print(f"Tracking fps: {fps:.1f}")
            frame_id += 1

    def _recognition_loop(self):
        """
        Match tracked boxes to detections and recognize them (background thread).
        """
        while not self._stop.is_set():
            with self._lock:
                raw_image = self.data_mapping["raw_image"]
                detection_bboxes = self.data_mapping["detection_bboxes"]
                detection_landmarks = self.data_mapping["detection_landmarks"]
                tracking_ids = list(self.data_mapping["tracking_ids"])
                tracking_bboxes = list(self.data_mapping["tracking_bboxes"])

            if raw_image is None or detection_landmarks is None or len(tracking_bboxes) == 0:
                time.sleep(0.005)
                continue

            h_img, w_img = raw_image.shape[:2]

            for track_id, track_bbox in zip(tracking_ids, tracking_bboxes):
                now = time.time()

                # 1) confidently identified already -> never re-run the recognizer
                with self._lock:
                    current = self.id_face_mapping.get(track_id)
                if current is not None and current != "UNKNOWN":
                    continue

                # 2) assign this track to its best-overlapping detection
                best_iou, best_j = 0.0, -1
                for j in range(len(detection_bboxes)):
                    iou = self._mapping_bbox(track_bbox, detection_bboxes[j])
                    if iou > best_iou:
                        best_iou, best_j = iou, j
                if best_j < 0 or best_iou < self.match_iou_threshold:
                    continue

                # 3) quality gate: skip partial (frame-edge) or tiny faces and wait
                #    for a clean crop, so we never commit a wrong UNKNOWN on a bad read
                x1, y1, x2, y2 = detection_bboxes[best_j][:4]
                if (x1 < self.edge_margin or y1 < self.edge_margin
                        or x2 > w_img - self.edge_margin or y2 > h_img - self.edge_margin):
                    continue

                # 4) adaptive throttle: retry an UNKNOWN track quickly for its first few
                #    attempts (recover from a bad entry crop), then back off to slow polling
                attempts = self._attempts.get(track_id, 0)
                interval = (
                    self.unknown_fast_interval if attempts < self.unknown_fast_retries
                    else self.unknown_retry_interval
                )
                if current == "UNKNOWN" and (now - self._last_attempt.get(track_id, 0.0)) < interval:
                    continue
                self._last_attempt[track_id] = now
                self._attempts[track_id] = attempts + 1

                # 5) align the matched detection and recognize it
                face_align = align_face(raw_image, detection_landmarks[best_j])
                score, name = self._recognize_face(face_align)

                if name is not None and score >= self.similarity_threshold:
                    caption = f"{name}:{score:.2f}"
                else:
                    caption = "UNKNOWN"

                with self._lock:
                    self.id_face_mapping[track_id] = caption

                detection_bboxes = np.delete(detection_bboxes, best_j, axis=0)
                detection_landmarks = np.delete(detection_landmarks, best_j, axis=0)
            time.sleep(0.005)

    def run(self, source: int=0):
        """
        Start the worker threads and run the display loop on the main thread.

        :param source: OpenCV VideoCapture source (camera index or path).
        """
        threading.Thread(target=self._capture_loop, args=(source,), daemon=True).start()
        threading.Thread(target=self._tracking_loop, daemon=True).start()
        threading.Thread(target=self._recognition_loop, daemon=True).start()

        try:
            # OpenCV GUI must run on the main thread
            while not self._stop.is_set():
                with self._lock:
                    frame = self.data_mapping["tracking_image"]
                if frame is not None:
                    cv2.imshow("Face Recognition", frame)
                if cv2.waitKey(1) in (27, ord("q"), ord("Q")):
                    break
        finally:
            self._stop.set()
            cv2.destroyAllWindows()


def main():
    """Build models, load the feature store, and run threaded recognition."""
    config = load_config(CONFIG_PATH)
    detector, recognizer = build_models(config)

    features = read_features(FEATURES_PATH)
    if features is None:
        raise FileNotFoundError(
            f"No feature store at {FEATURES_PATH}. Run app/add_persons_onnx.py first."
        )
    image_names, image_embeddings = features

    app = FaceRecognizer(config, detector, recognizer, image_names, image_embeddings)
    app.run(source=0)


if __name__ == "__main__":
    main()