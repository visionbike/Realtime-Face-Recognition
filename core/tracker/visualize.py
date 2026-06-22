from collections.abc import Sequence
import cv2
import numpy as np


_COLORS = (
    np.array(
        [
            0.000, 0.447, 0.741,
            0.850, 0.325, 0.098,
            0.929, 0.694, 0.125,
            0.494, 0.184, 0.556,
            0.466, 0.674, 0.188,
            0.301, 0.745, 0.933,
            0.635, 0.078, 0.184,
            0.300, 0.300, 0.300,
            0.600, 0.600, 0.600,
            1.000, 0.000, 0.000,
            1.000, 0.500, 0.000,
            0.749, 0.749, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 1.000,
            0.667, 0.000, 1.000,
            0.333, 0.333, 0.000,
            0.333, 0.667, 0.000,
            0.333, 1.000, 0.000,
            0.667, 0.333, 0.000,
            0.667, 0.667, 0.000,
            0.667, 1.000, 0.000,
            1.000, 0.333, 0.000,
            1.000, 0.667, 0.000,
            1.000, 1.000, 0.000,
            0.000, 0.333, 0.500,
            0.000, 0.667, 0.500,
            0.000, 1.000, 0.500,
            0.333, 0.000, 0.500,
            0.333, 0.333, 0.500,
            0.333, 0.667, 0.500,
            0.333, 1.000, 0.500,
            0.667, 0.000, 0.500,
            0.667, 0.333, 0.500,
            0.667, 0.667, 0.500,
            0.667, 1.000, 0.500,
            1.000, 0.000, 0.500,
            1.000, 0.333, 0.500,
            1.000, 0.667, 0.500,
            1.000, 1.000, 0.500,
            0.000, 0.333, 1.000,
            0.000, 0.667, 1.000,
            0.000, 1.000, 1.000,
            0.333, 0.000, 1.000,
            0.333, 0.333, 1.000,
            0.333, 0.667, 1.000,
            0.333, 1.000, 1.000,
            0.667, 0.000, 1.000,
            0.667, 0.333, 1.000,
            0.667, 0.667, 1.000,
            0.667, 1.000, 1.000,
            1.000, 0.000, 1.000,
            1.000, 0.333, 1.000,
            1.000, 0.667, 1.000,
            0.333, 0.000, 0.000,
            0.500, 0.000, 0.000,
            0.667, 0.000, 0.000,
            0.833, 0.000, 0.000,
            1.000, 0.000, 0.000,
            0.000, 0.167, 0.000,
            0.000, 0.333, 0.000,
            0.000, 0.500, 0.000,
            0.000, 0.667, 0.000,
            0.000, 0.833, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 0.167,
            0.000, 0.000, 0.333,
            0.000, 0.000, 0.500,
            0.000, 0.000, 0.667,
            0.000, 0.000, 0.833,
            0.000, 0.000, 1.000,
            0.000, 0.000, 0.000,
            0.143, 0.143, 0.143,
            0.286, 0.286, 0.286,
            0.429, 0.429, 0.429,
            0.571, 0.571, 0.571,
            0.714, 0.714, 0.714,
            0.857, 0.857, 0.857,
            0.000, 0.447, 0.741,
            0.314, 0.717, 0.741,
            0.50, 0.5, 0,
        ]
    )
    .astype(np.float32)
    .reshape(-1, 3)
)


def draw_detections(
        img: np.ndarray,
        boxes: np.ndarray,
        scores: np.ndarray,
        cls_ids: np.ndarray,
        conf: float = 0.5,
        class_names: list[str] | None = None
) -> np.ndarray:
    """
    Draw class detection boxes with labels on an image.

    :param img: BGR image to draw on (modified in place).
    :param boxes: Boxes of shape (N, 4) in (x1, y1, x2, y2) format.
    :param scores: Confidence scores of shape (N,).
    :param cls_ids: Class ids of shape (N,).
    :param conf: Minimum score required to draw a box.
    :param class_names: Sequence of class names indexed by class id.
    :return: The annotated image.
    """
    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = f"{class_names[cls_id]}:{(score * 100):.1f}%"
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
            txt_bk_color,
            -1,
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img


def get_color(idx: int) -> tuple[int, int, int]:
    """
    Generate a deterministic BGR color from a track index.

    :param idx: Track index.
    :return: BGR color tuple.
    """
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color


def plot_tracking(
        image: np.ndarray,
        tlwhs: Sequence,
        obj_ids: Sequence,
        frame_id: int = 0,
        fps: float = 0.0,
        ids2: Sequence | None = None,
        names: dict | None = None,
) -> np.ndarray:
    """
    Draw tracked bounding boxes with their ids on an image.

    :param image: Input BGR image.
    :param tlwhs: Boxes as (top-left x, top-left y, width, height).
    :param obj_ids: Track ids aligned with tlwhs.
    :param frame_id: Current frame index, shown in the overlay.
    :param fps: Frames-per-second value, shown in the overlay.
    :param ids2: Optional secondary ids drawn next to the main id.
    :param names: Optional mapping from track id to a display name.
    :return: The annotated image.
    """
    names = names or {}
    im = np.ascontiguousarray(np.copy(image))

    text_scale = 2
    text_thickness = 2
    line_thickness = 3

    cv2.putText(
        im,
        f"frame: {frame_id} fps: {fps:.2f} num: {len(tlwhs)}",
        (0, int(15 * text_scale)),
        cv2.FONT_HERSHEY_PLAIN,
        2,
        (0, 0, 255),
        thickness=2,
    )

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        int_box = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        # id_text = f"{int(obj_id)}"
        id_text = ""
        if obj_id in names:
            # id_text = id_text + ": " + names[obj_id]
            id_text = names[obj_id]
        if ids2 is not None:
            id_text = id_text + ", {}".format(int(ids2[i]))
        color = get_color(abs(obj_id))
        cv2.rectangle(im, int_box[0: 2], int_box[2: 4], color=color, thickness=line_thickness)
        cv2.putText(
            im,
            id_text,
            (int_box[0], int_box[1]),
            cv2.FONT_HERSHEY_PLAIN,
            text_scale,
            (0, 0, 255),
            thickness=text_thickness,
        )
    return im
