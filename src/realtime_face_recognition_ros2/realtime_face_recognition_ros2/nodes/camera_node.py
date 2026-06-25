

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class CameraNode(Node):
    def __init__(self):
        super().__init__("camera_node")
        # video_source is a string: "0" -> camera index 0, or a file/RTSP path
        self.declare_parameter("video_source", "0")
        self.declare_parameter("image_topic", "/image_raw")
        self.declare_parameter("fps", 30.0)
        self.declare_parameter("frame_id", "camera")

        video_src = str(self.get_parameter("video_source").value)
        topic = self.get_parameter("image_topic").value
        fps = float(self.get_parameter("fps").value)
        self.frame_id = self.get_parameter("frame_id").value

        if video_src.isdigit():
            self.cap = cv2.VideoCapture(int(video_src))
        else:
            self.cap = cv2.VideoCapture(video_src)
        if not self.cap.isOpened():
            self.get_logger().fatal(f"Cannot open video source: {video_src!r}")
            raise SystemExit(1)

        self.bridge = CvBridge()
        self.pub = self.create_publisher(Image, topic, 10)
        self.timer = self.create_timer(1.0 / max(fps, 1e-3), self._tick)
        self.get_logger().info(f"camera_node up: source={video_src!r} -> '{topic}' @ {fps:.1f} fps")

    def _tick(self):
        ok, frame = self.cap.read()
        if not ok:
            self.get_logger().warning("Frame grab failed (end of stream?); stopping.")
            self.timer.cancel()
            return
        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        self.pub.publish(msg)

    def destroy_node(self):
        if getattr(self, "cap", None) is not None:
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
