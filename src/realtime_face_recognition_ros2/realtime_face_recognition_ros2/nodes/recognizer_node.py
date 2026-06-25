import time
from typing import cast
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from cv_bridge import CvBridge
from realtime_face_recognition_ros2.core.constants import TRACKING_INPUT_SIZE
from realtime_face_recognition_ros2.core.aligner.alignment import align_face
from realtime_face_recognition_ros2.core.detector.scrfd_onnx import ScrfdONNX
from realtime_face_recognition_ros2.core.recognizer.arcface_onnx import ArcFaceONNX
from realtime_face_recognition_ros2.core.recognizer.feature_store import read_features, compare_embeddings
from realtime_face_recognition_ros2.core.tracker.byte_tracker import BYTETracker


class RecognizerNode(Node):
    def __init__(self):
        super().__init__("recognizer_node")

        # ---- parameters (paths default empty -> must be set via params.yaml) ----
        self.declare_parameter("detector_weights", "")
        self.declare_parameter("recognizer_weights", "")
        self.declare_parameter("features_path", "")
        self.declare_parameter("device_id", -1)
        self.declare_parameter("image_topic", "/image_raw")
        self.declare_parameter("results_topic", "/face_recognition/results")
        # detector
        self.declare_parameter("conf_threshold", 0.5)
        self.declare_parameter("nms_threshold", 0.4)
        # tracker
        self.declare_parameter("track_threshold", 0.5)
        self.declare_parameter("match_threshold", 0.8)
        self.declare_parameter("dup_iou_dist_threshold", 0.15)
        self.declare_parameter("aspect_ratio_threshold", 1.6)
        self.declare_parameter("min_box_area", 10.0)
        # recognizer
        self.declare_parameter("similarity_threshold", 0.5)
        self.declare_parameter("match_iou_threshold", 0.8)
        self.declare_parameter("unknown_retry_interval", 5.0)
        self.declare_parameter("unknown_fast_interval", 0.01)
        self.declare_parameter("unknown_fast_retries", 3)
        self.declare_parameter("min_face_size", 60.0)
        self.declare_parameter("edge_margin", 8.0)

        device_id = int(self.get_parameter("device_id").value)
        det_weights = self.get_parameter("detector_weights").value
        rec_weights = self.get_parameter("recognizer_weights").value
        feat_path = self.get_parameter("features_path").value

        if not det_weights or not rec_weights or not feat_path:
            self.get_logger().fatal(
                "detector_weights / recognizer_weights / features_path must be set (see config/params.yaml)."
            )
            raise SystemExit(1)

        # recognizer thresholds (instance state)
        self.similarity_threshold = float(self.get_parameter("similarity_threshold").value)
        self.match_iou_threshold = float(self.get_parameter("match_iou_threshold").value)
        self.unknown_retry_interval = float(self.get_parameter("unknown_retry_interval").value)
        self.unknown_fast_interval = float(self.get_parameter("unknown_fast_interval").value)
        self.unknown_fast_retries = int(self.get_parameter("unknown_fast_retries").value)
        self.min_face_size = float(self.get_parameter("min_face_size").value)
        self.edge_margin = float(self.get_parameter("edge_margin").value)
        self.aspect_ratio_threshold = float(self.get_parameter("aspect_ratio_threshold").value)
        self.min_box_area = float(self.get_parameter("min_box_area").value)

        # ---- models ----
        self.detector = ScrfdONNX(
            model_file=str(det_weights),
            conf_threshold=float(self.get_parameter("conf_threshold").value),
            nms_threshold=float(self.get_parameter("nms_threshold").value)
        )
        self.detector.prepare(device_id)
        self.recognizer = ArcFaceONNX(model_file=str(rec_weights))
        self.recognizer.prepare(device_id)

        self.tracker = BYTETracker(
            track_threshold=float(self.get_parameter("track_threshold").value),
            match_threshold=float(self.get_parameter("match_threshold").value),
            dup_iou_dist_threshold=float(self.get_parameter("dup_iou_dist_threshold").value),
            frame_rate=30
        )

        features = read_features(str(feat_path))
        if features is None:
            self.get_logger().fatal(f"No feature store at {feat_path}. Run the enrollment tool first.")
            raise SystemExit(1)
        self.image_names, self.image_embeddings = features

        # ---- persistent per-track recognition cache (was thread-shared state) ----
        self.id_face_mapping: dict[int, str] = {}
        self._last_attempt: dict[int, float] = {}
        self._attempts: dict[int, int] = {}

        # ---- ROS I/O ----
        self.bridge = CvBridge()
        self.pub_res = self.create_publisher(Detection2DArray, self.get_parameter("results_topic").value, 10)
        self.sub = self.create_subscription(Image, self.get_parameter("image_topic").value, self.on_image, 10)
        self.get_logger().info(
            f"recognizer_node up: sub '{self.get_parameter("image_topic").value}' -> '{self.get_parameter("results_topic").value}' "
            f"(device_id={device_id}, {len(self.image_names)} enrolled)"
        )

    @staticmethod
    def _iou(box1: np.ndarray, box2: np.ndarray) -> float:
        x1 = cast(float, max(box1[0], box2[0]))
        y1 = cast(float, max(box1[1], box2[1]))
        x2 = cast(float, min(box1[2], box2[2]))
        y2 = cast(float, min(box1[3], box2[3]))
        inter = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
        a1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        a2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
        union = a1 + a2 - inter
        return inter / union if union > 0 else 0.0

    def on_image(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:  # noqa: BLE001
            self.get_logger().warning(f"cv_bridge failed: {e}")
            return

        outputs, img_info, det_bboxes, det_landmarks = self.detector.detect_tracking(
            image=frame, input_size=TRACKING_INPUT_SIZE
        )
        raw = img_info["raw_img"]
        img_h, img_w = raw.shape[:2]

        ids, t_bboxes = [], []
        if outputs is not None and len(outputs) > 0:
            for t in self.tracker.update(outputs, (img_info["height"], img_info["width"]), TRACKING_INPUT_SIZE):
                tlwh = t.tlwh
                vertical = tlwh[2] / tlwh[3] > self.aspect_ratio_threshold
                too_small = tlwh[2] < self.min_face_size or tlwh[3] < self.min_face_size
                if tlwh[2] * tlwh[3] > self.min_box_area and not vertical or not too_small:
                    t_bboxes.append([tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]])
                    ids.append(t.track_id)

        # prune cache to active tracks so the dicts don't grow without bound
        active = set(ids)
        self.id_face_mapping = {i: n for i, n in self.id_face_mapping.items() if i in active}
        self._last_attempt = {i: v for i, v in self._last_attempt.items() if i in active}
        self._attempts = {i: v for i, v in self._attempts.items() if i in active}

        self._recognize(raw, ids, t_bboxes, det_bboxes, det_landmarks, img_w, img_h)

        # headless: publish structured JSON results only (no annotated image).
        # carry the source frame's stamp so the viewer can sync boxes to frames
        self.pub_res.publish(self._build_detections(ids, t_bboxes, msg.header))

    def _recognize(self, raw, ids, t_bboxes, det_bboxes, det_landmarks, img_w, img_h):
        if det_landmarks is None or len(t_bboxes) == 0:
            return
        for track_id, t_bbox in zip(ids, t_bboxes):
            current = self.id_face_mapping.get(track_id)
            now = time.time()
            # 1) already confidently identified -> never re-run
            if current is not None and current != "UNKNOWN":
                continue
            # 2) best-overlapping detection
            best_iou, best_j = 0.0, -1
            for j in range(len(det_bboxes)):
                iou = self._iou(t_bbox, det_bboxes[j])
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_j < 0 or best_iou < self.match_iou_threshold:
                continue
            # 3) quality gate: skip partial (edge) faces
            x1, y1, x2, y2 = det_bboxes[best_j][:4]
            if (x1 < self.edge_margin or y1 < self.edge_margin
                    or x2 > img_w - self.edge_margin or y2 > img_h - self.edge_margin):
                continue
            # 4) adaptive throttle for UNKNOWN tracks
            attempts = self._attempts.get(track_id, 0)
            interval = (self.unknown_fast_interval if attempts < self.unknown_fast_retries
                        else self.unknown_retry_interval)
            if current == "UNKNOWN" and (now - self._last_attempt.get(track_id, 0.0)) < interval:
                continue
            self._last_attempt[track_id] = now
            self._attempts[track_id] = attempts + 1
            # 5) align + recognize
            face = align_face(raw, det_landmarks[best_j])
            emb = self.recognizer.get_feature(face)
            score, idx = compare_embeddings(emb, self.image_embeddings)
            name = self.image_names[idx]
            caption = f"{name}:{score:.2f}" if (name is not None and score >= self.similarity_threshold) else "UNKNOWN"
            self.id_face_mapping[track_id] = caption
            # consume the matched detection so two tracks can't claim it
            det_bboxes = np.delete(det_bboxes, best_j, axis=0)
            det_landmarks = np.delete(det_landmarks, best_j, axis=0)

    def _build_detections(self, ids, t_bboxes, header) -> Detection2DArray:
        arr = Detection2DArray()
        arr.header = header
        for track_id, bbox in zip(ids, t_bboxes):
            x1, y1, x2, y2 = bbox
            det = Detection2D()
            det.header = header
            det.id = str(int(track_id))
            det.bbox.center.position.x = (float(x1) + float(x2)) / 2.0
            det.bbox.center.position.y = (float(y1) + float(y2)) / 2.0
            det.bbox.size_x = float(x2 - x1)
            det.bbox.size_y = float(y2 - y1)
            # tri-state: pending -> no hypothesis; else UNKNOWN / recognized
            caption = self.id_face_mapping.get(track_id, "...")
            if caption != "...":
                hyp = ObjectHypothesisWithPose()
                if caption == "UNKNOWN":
                    hyp.hypothesis.class_id = "UNKNOWN"
                    hyp.hypothesis.score = 0.0
                else:
                    name, _, score = caption.partition(":")
                    hyp.hypothesis.class_id = name
                    hyp.hypothesis.score = float(score) if score else 0.0
                det.results.append(hyp)
            arr.detections.append(det)
        return arr


def main(args=None):
    rclpy.init(args=args)
    node = RecognizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
