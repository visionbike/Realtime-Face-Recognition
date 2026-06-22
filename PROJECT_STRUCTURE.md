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
