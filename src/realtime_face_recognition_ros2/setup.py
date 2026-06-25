import os
from glob import glob
from setuptools import setup, find_packages

package_name = "realtime_face_recognition_ros2"

setup(
    name=package_name,
    version="0.0.1",
    # finds: realtime_face_recognition_ros2 and all subpackages
    packages=find_packages(exclude=["tests"]),
    data_files=[
        # ament resource index marker
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        # package manifest
        (os.path.join("share", package_name), ["package.xml"]),
        # launch files
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        # config (thresholds + ROS params)
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="visionbike",
    maintainer_email="phuc.visionbike@gmail.com",
    description="Real-time face detection, tracking and recognition as ROS 2 nodes.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "camera_node = realtime_face_recognition_ros2.nodes.camera_node:main",
            "oakd_camera_node = realtime_face_recognition_ros2.nodes.oakd_camera_node:main",
            "recognizer_node = realtime_face_recognition_ros2.nodes.recognizer_node:main",
            "viewer_node = realtime_face_recognition_ros2.nodes.viewer_node:main"
        ]
    }
)