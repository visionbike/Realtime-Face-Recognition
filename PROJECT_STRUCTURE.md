# Project Structure

```
Realtime-Face-Recognition/
│
├── app/                            # Entry points only
│   ├── recognize.py                # Main real-time pipeline
│   └── add_persons.py              # Database enrollment tool
│
├── core/                           # All inference logic
│   ├── __init__.py
│   ├── detector/
│   │   ├── __init__.py
│   │   ├── scrfd.py                # SCRFD ONNX detector (primary)
│   │   └── yolov5.py               # YOLOv5 detector (alternative)
│   ├── recognizer/
│   │   ├── __init__.py
│   │   ├── arcface.py              # ArcFace model
│   │   └── feature_store.py        # Feature read/compare utils
│   ├── aligner/
│   │   ├── __init__.py
│   │   └── alignment.py            # Face alignment (ArcFace crop)
│   └── tracker/
│       ├── __init__.py
│       ├── byte_tracker.py
│       ├── kalman_filter.py
│       ├── matching.py
│       ├── basetrack.py
│       └── visualize.py
│
├── weights/                        # Model files — gitignored
│   ├── detection/
│   │   └── scrfd_2.5g_bnkps.onnx
│   └── recognition/
│       └── arcface_r100.pth
│
├── datasets/                       # Face database — gitignored
│   ├── new_persons/                # Drop images here to enroll
│   ├── backup/                     # Moved after enrollment
│   ├── faces/                      # Cropped face images
│   └── features/                   # feature.npz
│
├── configs/
│   └── tracking.yaml
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Source Mapping (from `face-recognition`)

| New path | Old path |
|---|---|
| `core/detector/scrfd.py` | `face_detection/scrfd/detector.py` |
| `core/detector/yolov5.py` | `face_detection/yolov5_face/detector.py` |
| `core/recognizer/arcface.py` | `face_recognition/arcface/model.py` |
| `core/recognizer/feature_store.py` | `face_recognition/arcface/utils.py` |
| `core/aligner/alignment.py` | `face_alignment/alignment.py` |
| `core/tracker/byte_tracker.py` | `face_tracking/tracker/byte_tracker.py` |
| `core/tracker/kalman_filter.py` | `face_tracking/tracker/kalman_filter.py` |
| `core/tracker/matching.py` | `face_tracking/tracker/matching.py` |
| `core/tracker/basetrack.py` | `face_tracking/tracker/basetrack.py` |
| `core/tracker/visualize.py` | `face_tracking/tracker/visualize.py` |
| `app/recognize.py` | `recognize.py` + `recognize2.py` (merged) |
| `app/add_persons.py` | `add_persons.py` |
| `configs/tracking.yaml` | `face_tracking/config/config_tracking.yaml` |
| `weights/detection/` | `face_detection/scrfd/weights/` |
| `weights/recognition/` | `face_recognition/arcface/weights/` |

## Dropped (not needed for inference)

| Path | Reason |
|---|---|
| `face_detection/retinaface/` | Training-only |
| `face_detection/yolov5_face/utils/` | Training utilities |
| `scrfd2onnx.py` | Model conversion tool |
| `detect.py`, `tracking.py`, `face_align.py` | Ad-hoc test scripts |
| `recognize2.py` | Merged into `app/recognize.py` |
| `face_tracking/pretrained/` | Unused |

---

# ROS 2 Package (branch `ros2-lyrical`)

A single `ament_python` package that **vendors the existing `core/`** and wraps the
pipeline as nodes. The repo root doubles as the colcon workspace root.

```
Realtime-Face-Recognition/                  # colcon workspace root (= repo root)
│
├── src/
│   └── face_recognition_ros/               # ament_python package
│       ├── package.xml                     # exec deps: rclpy, sensor_msgs, std_msgs, cv_bridge
│       ├── setup.py                         # entry_points -> nodes; find_packages(); ships config/ + launch/
│       ├── setup.cfg                        # script_dir = $base/lib/face_recognition_ros
│       ├── resource/
│       │   └── face_recognition_ros         # empty ament resource marker
│       │
│       ├── face_recognition_ros/            # importable package
│       │   ├── __init__.py
│       │   ├── core/                        # VENDORED from ./core (intra-core imports are
│       │   │   ├── detector/                #   relative, so they survive the move unchanged)
│       │   │   ├── recognizer/
│       │   │   ├── aligner/
│       │   │   └── tracker/
│       │   ├── pipeline.py                  # refactored from app/recognize_onnx.py:FaceRecognizer
│       │   │                                #   contract: process(frame, frame_id, fps)
│       │   │                                #       -> (annotated_frame, results[])
│       │   │                                #   no threads; per-track cache (id_face_mapping,
│       │   │                                #   _last_attempt, _attempts) kept as instance state
│       │   └── nodes/
│       │       ├── __init__.py
│       │       ├── recognizer_node.py       # sub /image_raw -> pipeline -> pub annotated Image + String(JSON)
│       │       └── camera_node.py           # optional: cv2.VideoCapture(source) -> pub /image_raw (test source)
│       │
│       ├── launch/
│       │   └── face_recognition.launch.py   # camera_node + recognizer_node, loads params.yaml
│       └── config/
│           ├── config.yaml                  # copy of cfgs/config.yaml (thresholds only — small, static)
│           └── params.yaml                  # ROS params: weights/features paths, device_id, topics, source
│
├── weights/                                 # NOT in the package — referenced by absolute param path
│   ├── detection/scrfd_2.5g_bnkps.onnx      #   (544 MB total; only the .onnx are runtime deps)
│   └── recognition/arcface_r100_int8.onnx
├── datasets/
│   └── features/                            # mutable runtime state (written by enrollment) — kept external
│
│   ── colcon build artifacts (gitignored) ──
├── build/   install/   log/
```

## Why `weights/` and `datasets/` stay outside the package

| Asset | Reason it stays external |
|---|---|
| `weights/` (544 MB) | `ament_python` copies `data_files` into `install/share/` on every build — would duplicate ~0.5 GB and slow rebuilds. Models are deployment assets, not source. |
| `datasets/features/` | Mutable state written by `add_persons_onnx.py` and read by the recognizer. `install/` is rewritten on rebuild, so an installed copy gets clobbered. |
| `*.pth` files | Training/export inputs only — runtime uses just the two `.onnx` (~128 MB). Not runtime deps. |

Only the small static `cfgs/config.yaml` is shipped inside the package (`config/config.yaml`).

## Nodes and topics

| Node | Subscribes | Publishes |
|---|---|---|
| `recognizer_node` | `/image_raw` (`sensor_msgs/Image`) | `/face_recognition/image_annotated` (`sensor_msgs/Image`), `/face_recognition/results` (`std_msgs/String`, JSON: `track_id, bbox, name, score`) |
| `camera_node` (test) | — | `/image_raw` (`sensor_msgs/Image`) |

First-pass interface is `std_msgs/String` JSON (no custom `.msg`, which would force a
second `ament_cmake`/`rosidl` package). Swap to `vision_msgs`-style messages later.

## Environment (Python must match ROS 2 lyrical = 3.14)

`rclpy`/`cv_bridge` ship compiled against `/usr/bin/python3.14`
(`_rclpy_pybind11.cpython-314-...so`), so the runtime interpreter must be 3.14.
Use a venv built from that same interpreter (lowest ABI risk vs. a conda mix):

```bash
/usr/bin/python3.14 -m venv ~/venvs/face-ros
source ~/venvs/face-ros/bin/activate
pip install -r requirements_cpu.txt          # all deps have cp314 x86_64 wheels (verified)

# every ROS session, in this order:
source /opt/ros/lyrical/setup.bash           # provides rclpy + cv_bridge (PYTHONPATH)
source ~/venvs/face-ros/bin/activate
```

## Build & test

```bash
cd ~/projects/Realtime-Face-Recognition
colcon build --symlink-install --packages-select face_recognition_ros
source install/setup.bash
ros2 launch face_recognition_ros face_recognition.launch.py
# or piecewise:
ros2 run face_recognition_ros camera_node
ros2 run face_recognition_ros recognizer_node
ros2 topic echo /face_recognition/results
ros2 run rqt_image_view rqt_image_view /face_recognition/image_annotated
```

## ROS source mapping

| ROS package path | From |
|---|---|
| `src/face_recognition_ros/face_recognition_ros/core/` | `core/` (vendored, imports unchanged) |
| `src/face_recognition_ros/face_recognition_ros/pipeline.py` | `app/recognize_onnx.py` (`FaceRecognizer`, de-threaded) |
| `src/face_recognition_ros/config/config.yaml` | `cfgs/config.yaml` |
| `weights/`, `datasets/features/` | unchanged at repo root, passed as params |
