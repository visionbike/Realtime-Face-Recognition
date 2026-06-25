from typing import cast
import depthai as dai
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class OakDCameraNode(Node):
    """
    Publishes the OAK-D RGB stream as sensor_msgs/Image on `image_topic`.

    Drop-in for camera_node: same topic/fps/frame_id contract, so the
    recognizer and viewer need no changes. Uses the DepthAI v3 API.
    """

    def __init__(self):
        super().__init__("oakd_camera_node")
        self.declare_parameter("image_topic", "/image_raw")
        self.declare_parameter("fps", 30.0)
        self.declare_parameter("frame_id", "oak_rgb_camera_optical_frame")
        self.declare_parameter("width", 1280)
        self.declare_parameter("height", 720)
        # optional: target a specific device by MxID/IP (empty -> first found)
        self.declare_parameter("device_id", "")

        topic = self.get_parameter("image_topic").value
        self.fps = float(self.get_parameter("fps").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        width = int(self.get_parameter("width").value)
        height = int(self.get_parameter("height").value)
        device_id = str(self.get_parameter("device_id").value)

        # ---- DepthAI pipeline: single RGB camera -> host output queue ----
        try:
            self.pipeline = (
                dai.Pipeline(dai.Device(dai.DeviceInfo(device_id))) if device_id
                else dai.Pipeline()
            )
            cam = self.pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
            cam_out = cam.requestOutput(
                (width, height), dai.ImgFrame.Type.BGR888i, fps=self.fps
            )
            # small, non-blocking queue: we always want the freshest frame
            self.queue = cam_out.createOutputQueue(maxSize=4, blocking=False)
            self.pipeline.start()
        except Exception as e:  # noqa: BLE001
            self.get_logger().fatal(f"Cannot start OAK-D pipeline: {e}")
            raise SystemExit(1)

        self.bridge = CvBridge()
        self.pub = self.create_publisher(Image, topic, 10)
        # poll a bit faster than the stream so we don't add latency
        self.timer = self.create_timer(1 / max(self.fps * 1.5, 1e-3), self._tick)
        self.get_logger().info(f"oakd_camera_node up: OAK-D RGB {width}x{height} -> '{topic}' @ {self.fps:.1f} fps")

    def _tick(self):
        if not self.pipeline.isRunning():
            self.get_logger().warning("OAK-D pipeline stopped; shutting down camera.")
            self.timer.cancel()
            return
        pkt = self.queue.tryGet()   # non-blocking: mat be None between frames
        if pkt is None:
            return
        frame = cast(dai.ImgFrame, pkt).getCvFrame()    # BGR np.ndarray for BGR888i
        msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        self.pub.publish(msg)

    def destroy(self):
        if self.pipeline is not None and self.pipeline.isRunning():
            self.pipeline.stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = OakDCameraNode()
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