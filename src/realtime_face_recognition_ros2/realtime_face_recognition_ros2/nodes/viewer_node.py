import time
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge
from realtime_face_recognition_ros2.core.tracker.visualize import get_color


class ViewerNode(Node):
    """
    Display node for the headless recognizer.

    The recognizer no longer emits an annotated image; it publishes a
    Detection2DArray whose header stamp matches the source frame. This node
    buffers raw frames keyed by that stamp and, when the matching detections
    arrive, draws the boxes/captions itself and shows the synced frame.
    """

    def __init__(self):
        super().__init__("viewer_node")
        self.declare_parameter("image_topic", "/image_raw")
        self.declare_parameter("results_topic", "/face_recognition/results")
        self.declare_parameter("window_name", "Face Recognition")
        # cap on buffered frames so a stalled recognizer can't grow memory
        self.declare_parameter("frame_buffer", 60)

        self.window = str(self.get_parameter("window_name").value)
        image_topic = self.get_parameter("image_topic").value
        results_topic = self.get_parameter("results_topic").value
        self._buffer_size = int(self.get_parameter("frame_buffer").value)

        self.bridge = CvBridge()
        # stamp (ns) -> BGR frame; insertion-ordered (py3.7+) so we can age out
        self._frames: dict[int, "cv2.typing.MatLike"] = {}

        # display-fps bookkeeping (frames shown per second, recomputed every 30)
        self._t0 = time.time_ns()
        self._fcount = 0
        self._fps = 0.0
        # resize the window to the camera frame size on the first displayed frame
        self._win_size: tuple[int, int] | None = None

        self.sub_img = self.create_subscription(Image, image_topic, self.on_image, 10)
        self.sub_res = self.create_subscription(Detection2DArray, results_topic, self.on_results, 10)
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        self.get_logger().info(
            f"viewer_node up: frames '{image_topic}' + results '{results_topic}' (press q/ESC to quit)"
        )

    @staticmethod
    def _stamp_ns(header) -> int:
        return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)

    def on_image(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:  # noqa: BLE001
            self.get_logger().warning(f"cv_bridge failed: {e}")
            return
        self._frames[self._stamp_ns(msg.header)] = frame
        # drop oldest frames if the recognizer falls behind
        while len(self._frames) > self._buffer_size:
            self._frames.pop(next(iter(self._frames)))

    def on_results(self, msg: Detection2DArray):
        stamp = self._stamp_ns(msg.header)
        frame = self._frames.pop(stamp, None)
        if frame is None:
            # frame already aged out (or never seen); nothing to draw on
            return
        # discard every buffered frame at or before the one we just consumed,
        # so the buffer only holds frames newer than the latest drawn result
        for older in [s for s in self._frames if s <= stamp]:
            self._frames.pop(older)

        self._tick_fps()
        self._draw(frame, msg)
        # match the window to the camera frame size (once, or if it changes)
        h, w = frame.shape[:2]
        if self._win_size != (w, h):
            cv2.resizeWindow(self.window, w, h)
            self._win_size = (w, h)
        cv2.imshow(self.window, frame)
        # waitKey pumps the GUI event loop; runs in the spin (main) thread,
        # so imshow here is safe with the default single-threaded executor.
        if cv2.waitKey(1) in (27, ord("q"), ord("Q")):
            self.get_logger().info("Quit key pressed; shutting down.")
            rclpy.shutdown()

    def _draw(self, frame, msg: Detection2DArray):
        font = cv2.FONT_HERSHEY_PLAIN
        for det in msg.detections:
            cx = det.bbox.center.position.x
            cy = det.bbox.center.position.y
            w = det.bbox.size_x
            h = det.bbox.size_y
            x1 = int(cx - w / 2.0)
            y1 = int(cy - h / 2.0)
            x2 = int(cx + w / 2.0)
            y2 = int(cy + h / 2.0)

            track_id = int(det.id) if det.id else 0
            # tri-state caption: no hypothesis -> pending; else UNKNOWN / name:score
            if det.results:
                hyp = det.results[0].hypothesis
                caption = hyp.class_id if hyp.class_id == "UNKNOWN" else f"{hyp.class_id}:{hyp.score:.2f}"
            else:
                caption = "..."

            color = get_color(abs(track_id))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color=color, thickness=3)
            (tw, th), baseline = cv2.getTextSize(caption, font, 1.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - baseline), (x1 + tw, y1 + baseline), (0, 0, 0), -1)
            cv2.putText(frame, caption, (x1, y1), font, 1.5, (255, 255, 255), thickness=1)

        overlay = f"Fps: {self._fps:.2f} Num: {len(msg.detections)}"
        (tw, th), baseline = cv2.getTextSize(overlay, font, 1.0, 1)
        cv2.rectangle(frame, (0, 0), (tw, th + baseline + 4), (0, 0, 0), -1)
        cv2.putText(frame, overlay, (0, th + 2), font, 1.0, (255, 255, 255), thickness=1)

    def _tick_fps(self):
        self._fcount += 1
        if self._fcount >= 30:
            self._fps = 1e9 * self._fcount / (time.time_ns() - self._t0)
            self._fcount = 0
            self._t0 = time.time_ns()

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ViewerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():      # callback may have already called shutdown()
            rclpy.shutdown()

if __name__ == "__main__":
    main()
