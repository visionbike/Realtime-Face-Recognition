import sys
import argparse
import shutil
from pathlib import Path
import cv2
import numpy as np
import yaml
from tqdm import tqdm

# make core importable when this file is run as app/add_persons.py
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src" / "realtime_face_recognition_ros2" / "realtime_face_recognition_ros2"))

from core.constants import IMAGE_EXTS
from core.detector.scrfd_onnx import ScrfdONNX
from core.recognizer.arcface_onnx import ArcFaceONNX
from core.recognizer.feature_store import read_features
from core.aligner.alignment import align_face

# ONNX Runtime device id: negative -> CPU, >=0 -> CUDA device index
DEVICE_ID = -1

# asset locations
CONFIG_PATH = ROOT_DIR / "cfgs" / "config.yaml"
DETECTOR_WEIGHTS = ROOT_DIR / "weights" / "detection" / "scrfd_2.5g_bnkps.onnx"
RECOGNIZER_WEIGHTS = ROOT_DIR / "weights" / "recognition" / "arcface_r100_int8.onnx"


def load_config(config_path: Path) -> dict:
    """
    Load the YAML config; return an empty dict if no config is found.

    :param config_path: Path to the config file.
    :return: Parsed configuration as a dict or an empty dict if no config is found.
    """
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_models(config: dict) -> tuple[ScrfdONNX, ArcFaceONNX]:
    """
    Instantiate the SCRFD detector and the ArcFace recognizer.

    :param config: Parsed config dict
    :return: Initialized face detector and recognizer.
    """
    det_cfg = config.get("detector", {})
    detector = ScrfdONNX(
        model_file=str(DETECTOR_WEIGHTS),
        conf_threshold=det_cfg.get("conf_threshold", 0.5),
        nms_threshold=det_cfg.get("nms_threshold", 0.4)
    )
    detector.prepare(DEVICE_ID)
    recognizer = ArcFaceONNX(model_file=str(RECOGNIZER_WEIGHTS))
    recognizer.prepare(DEVICE_ID)
    return detector, recognizer


def add_persons(
    backup_dir: str,
    add_persons_dir: str,
    faces_save_dir: str,
    features_dir: str,
):
    """
    Enroll every person folder found under add_persions_path.

    :param backup_dir: Directory enrolled source folders are moved to.
    :param add_persons_dir: Directory holding <person_name>/<image> to enroll.
    :param faces_save_dir: Directory where aligned face crops are saved.
    :param features_dir: Feature store path.
    """
    config = load_config(CONFIG_PATH)
    detector, recognizer = build_models(config)

    add_persons_dir = Path(add_persons_dir)
    faces_save_dir = Path(faces_save_dir)
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    features_dir = Path(features_dir)
    features_dir.mkdir(parents=True, exist_ok=True)

    image_names: list[str] = []
    image_embeddings: list[np.ndarray] = []

    for person_path in sorted(p for p in add_persons_dir.iterdir() if p.is_dir()):
        name_person = person_path.name
        person_face_path = faces_save_dir / name_person
        person_face_path.mkdir(parents=True, exist_ok=True)
        image_files = [p for p in person_path.iterdir() if p.suffix.lower() in IMAGE_EXTS]
        for image_file in tqdm(image_files, desc=f"Enrolling {name_person}"):
            input_image = cv2.imread(str(image_file))
            if input_image is None:
                continue

            # SCRFD now returns (bboxes[N, 5], landmarks[N, 5, 2]; use landmarks for alignment
            bboxes, landmarks = detector.detect(input_image)
            if bboxes is None or len(bboxes) == 0 or landmarks is None:
                continue

            for i in range(len(bboxes)):
                # align with the 5-point landmarks instead of a raw bbox crop
                face_image = align_face(input_image, landmarks[i].astype(np.float32))

                number_files = len(list(person_face_path.glob("*.jpg")))
                cv2.imwrite(str(person_face_path / f"{person_path.name}_{number_files}.jpg"), face_image)

                # ArcFaceONNX takes the BGR crop directly (blobFromImage handles BGR->RGB + scaling)
                image_embeddings.append(recognizer.get_feature(face_image))
                image_names.append(name_person)

    if not image_embeddings:
        print("No face images found.")
        return

    array_embeddings = np.array(image_embeddings)
    array_names = np.array(image_names)

    # merge with existing store. Note the upgraded key names
    features = read_features(features_dir)
    if features is not None:
        old_image_names, old_image_embeddings = features
        array_names = np.hstack((old_image_names, array_names))
        array_embeddings = np.vstack((old_image_embeddings, array_embeddings))
        print("Updating existing features!")

    np.savez_compressed(
        features_dir / "features",
        image_name=array_names,
        image_embeddings=array_embeddings
    )

    # Move enrolled source folders into backup
    for sub_dir in add_persons_dir.iterdir():
        if sub_dir.is_dir() and not (backup_dir / sub_dir.name).exists():
            shutil.move(str(sub_dir), str(backup_dir))
    print(f"Successfully added {len(image_names)} images.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enroll new persons into the face database.")
    parser.add_argument("--backup-dir", type=str, default="./datasets/backup", help="Directory to move enrolled source folders into.")
    parser.add_argument("--add-persons-dir", type=str, default="./datasets/new_persons", help="Directory containing <person_name>/<images> to enroll.")
    parser.add_argument("--faces-save-dir", type=str, default="./datasets/faces", help="Directory to save aligned face crops.")
    parser.add_argument("--features-dir", type=str, default="./datasets/features", help="Feature store directory, without the .npz extension.")
    opt = parser.parse_args()

    add_persons(**vars(opt))
