import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, EmitEvent
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.events import Shutdown
from launch_ros.actions import Node


PACKAGE = "realtime_face_recognition_ros2"


def generate_launch_description():
    share = get_package_share_directory(PACKAGE)
    config = os.path.join(share, "config", "config.yaml")   # thresholds
    params = os.path.join(share, "config", "params.yaml")   # paths/ topics/ source

    camera = LaunchConfiguration("camera")
    use_viewer = LaunchConfiguration("use_viewer")

    # condition helpers: which camera backend to bring up
    is_oakd = IfCondition(PythonExpression(["'", camera, "' == 'oakd'"]))
    is_video = IfCondition(PythonExpression(["'", camera, "' == 'video'"]))

    # heep a handle on the viewer so we can watch for it exit
    viewer_node = Node(
        package=PACKAGE,
        executable="viewer_node",
        name="viewer_node",
        parameters=[params],
        output="screen",
        condition=IfCondition(use_viewer)
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "camera",
            default_value="oakd",
            choices=["oakd", "video"],
            description="Camera backend 'oakd' (OAK-D via depthai) or 'video' (cv2.VideoCapture)."
        ),
        DeclareLaunchArgument(
            "use_viewer",
            default_value="true",
            description="Launch the OpenCV viewer window (set false for headless hosts."
        ),

        # --  camera (exactly one runs, chosen by 'camera:=') ---
        Node(
            package=PACKAGE,
            executable="oakd_camera_node",
            name="oakd_camera_node",
            parameters=[params],
            output="screen",
            condition=is_oakd
        ),
        Node(
            package=PACKAGE,
            executable="camera_node",
            name="camera_node",
            parameters=[params],
            output="screen",
            condition=is_video
        ),

        # --- recognizer ---
        Node(
            package=PACKAGE,
            executable="recognizer_node",
            name="recognizer_node",
            parameters=[config, params],
            output="screen"
        ),

        # --- viewer ---
        viewer_node,

        # when the viewer exits (q/ESC), shut the whole launch down
        RegisterEventHandler(
            OnProcessExit(
                target_action=viewer_node,
                on_exit=[EmitEvent(event=Shutdown(reason="viewer closed"))]
            )
        )
    ])