from collections.abc import Sequence
from typing import Literal, cast
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from .strack import STrack


def linear_assignment(
        cost_matrix: np.ndarray,
        threshold: float
) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]:
    """
    Solve the linear assignment problem and filter matches by a cost threshold.

    :param cost_matrix: Cost matrix of shape (N, M).
    :param threshold: Maximum allowed cost for a valid match.
    :return: Tuple of (matches, unmatched_a, unmatched_b), where
        matches: Array of [row, col] pairs
        unmatched_a, unmatched_b: the unmatched entries.
    """
    if cost_matrix.size == 0:
        return (
          np.empty((0, 2), dtype=int),
          tuple(range(cost_matrix.shape[0])),
          tuple(range(cost_matrix.shape[1])),
        )

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches = np.array([[r, c] for r, c in zip(row_ind, col_ind) if cost_matrix[r, c] <= threshold])
    unmatched_a = np.array([i for i in range(cost_matrix.shape[0]) if i not in row_ind])
    unmatched_b = np.array([i for i in range(cost_matrix.shape[1]) if i not in col_ind])

    return matches, tuple(unmatched_a), tuple(unmatched_b)


def bbox_iou(tlbr1: np.ndarray, tlbr2: np.ndarray) -> float:
    """
    Compute the IoU of two bounding boxes in (top-left x, top-left y, bottom-right x, bottom-right y) format.

    :param tlbr1: First box of shape (4,), format (xtl_1, ytl_1, xbr_1, ybr_1).
    :param tlbr2: Second box of shape (4,), format (xtl_2, ytl_2, xbr_2, ybr_2).
    :return: Intersection-over-union value.
    """
    xtl_1, ytl_1, xbr_1, ybr_1 = map(float, tlbr1)
    xtl_2, ytl_2, xbr_2, ybr_2 = map(float, tlbr2)

    # Calculate the area of the intersection rectangle.
    xtl = max(xtl_1, xtl_2)
    ytl = max(ytl_1, ytl_2)
    xbr = min(xbr_1, xbr_2)
    ybr = min(ybr_1, ybr_2)
    inter_area = max(xbr - xtl, 0.0) * max(ybr - ytl, 0.0)

    # Calculate each box area.
    box1_area = (xbr_1 - xtl_1) * (ybr_1 - ytl_1)
    box2_area = (xbr_2 - xtl_2) * (ybr_2 - ytl_2)

    # Calculate union area and IoU.
    union_area = box1_area + box2_area - inter_area
    if union_area <= 0.0:
        return 0.0
    iou = inter_area / union_area

    return iou


def ious(tlbrs1: Sequence[np.ndarray], tlbrs2: Sequence[np.ndarray]) -> np.ndarray:
    """
    Compute the IoU matrix between two sets of boxes in (top-left x, top-left y, bottom-right x, bottom-right y) format.

    :param tlbrs1: First set of boxes as a sequence of (xtl_1, ytl_1, xbr_1, ybr_1) arrays of shape (4,).
    :param tlbrs2: Second set of boxes as a sequence of (xtl_2, ytl_2, xbr_2, ybr_2) arrays of shape (4,).
    :return: IoU matrix of shape (len(tlbrs1), len(tlbrs1)).
    """
    ious_matrix = np.zeros((len(tlbrs1), len(tlbrs2)), dtype=np.float64)
    for i, tlbr1 in enumerate(tlbrs1):
        for j, tlbr2 in enumerate(tlbrs2):
            ious_matrix[i, j] = bbox_iou(tlbr1, tlbr2)
    return ious_matrix


def iou_distance(tracks1: list[STrack] | np.ndarray, tracks2: list[STrack] | np.ndarray) -> np.ndarray:
    """
    Compute an IoU-based cost matrix between two sets of tracks.

    :param tracks1: First set of the list of STrack or an array of (x1, y1, x2, y2) boxes.
    :param tracks2: Second set of the list of STrack or an array of (x1, y1, x2, y2) boxes.
    :return: Cost matrix (1 - IoU) of shape (len(tracks1), len(tracks2)).
    """
    if (len(tracks1) > 0 and isinstance(tracks1[0], np.ndarray)) or (
            len(tracks2) > 0 and isinstance(tracks2[0], np.ndarray)
    ):
        tlbrs1 = cast(Sequence[np.ndarray], tracks1)
        tlbrs2 = cast(Sequence[np.ndarray], tracks2)
    else:
        tlbrs1 = [track.tlbr for track in tracks1]
        tlbrs2 = [track.tlbr for track in tracks2]
    _ious = ious(tlbrs1, tlbrs2)
    cost_matrix = 1 - _ious

    return cost_matrix


def embedding_distance(
        tracks: list[STrack] | np.ndarray,
        detections: list[STrack] | np.ndarray,
        metric: Literal["cosine", "euclidean"]  = "cosine"
) -> np.ndarray:
    """
    Compute an appearance-based cost matrix between tracks and detections.

    Each track's smoothed (EMA) ReID feature is compared against each detection's current feature
    using the given distance metric. Features are assumed to be L2-normalized.
    Distances are clamped at 0 so the cost stays non-negative.

    :param tracks: Existing tracks, each exposing a smooth_feat of shape (D,).
    :param detections: New detections, each exposing a curr_feat of shape (D,).
    :param metric: Distance metric, e.g. "cosine" or "euclidean".
    :return: Cost matrix of shape (len(tracks), len(detections)).
    """
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float64)
    if cost_matrix.size == 0:
        return cost_matrix
    if any(t.curr_feat is None for t in detections) or any(t.smooth_feat is None for t in tracks):
        return cost_matrix  # already all-zero; no appearance info available
    det_features = np.asarray([t.curr_feat for t in detections], dtype=np.float64)
    track_features = np.asarray([t.smooth_feat for t in tracks], dtype=np.float64)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))
    return cost_matrix


def fuse_score(cost_matrix: np.ndarray, detections: list[STrack]) -> np.ndarray:
    """
    Fuse an IoU cost matrix with detection confidence scores.

    Converts the IoU cost to a similarity (1 - cost), multiplies each column by its
    detection's confidence score so high-confidence detections are favored, then
    converts back to a cost (1 - fused similarity).

    :param cost_matrix: IoU cost matrix of shape (len(tracks), len(detections)).
    :param detections: Detections aligned with the columns of cost_matrix, each exposing a score.
    :return: Score-fused cost matrix of the same shape; returned unchanged if empty.
    """
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    return 1 - fuse_sim


def fuse_appearance(
    emb_dist: np.ndarray,
    iou_dist: np.ndarray,
    detections: list[STrack],
    appearance_threshold: float = 0.25,
    proximity_threshold: float = 0.5,
) -> np.ndarray:
    """
    Combine appearance and motion costs into a single association cost.

    The raw IoU cost masks geometrically implausible pairs. Appearance costs above
    appearance_thresh, or whose IoU cost exceeds proximity_thresh, are rejected
    (set to 1.0). The IoU cost is score-fused via fuse_score, and the final cost is the
    elementwise minimum of the score-fused IoU cost and the gated appearance cost, so a
    strong appearance match can recover a track despite weak box overlap.

    :param emb_dist: Appearance cost matrix from embedding_distance, shape (T, D).
    :param iou_dist: Raw IoU cost matrix (1 - IoU), shape (T, D).
    :param detections: Detections aligned with the columns, each exposing a score.
    :param appearance_threshold: Max appearance cost to accept a match.
    :param proximity_threshold: Max IoU cost (1 - IoU) for a pair to stay eligible.
    :return: Fused cost matrix of shape (T, D); falls back to fuse_score(iou_dist) if no embeddings.
    """
    iou_fused = fuse_score(iou_dist, detections)
    if emb_dist.size == 0:
        return iou_fused
    emb = emb_dist.copy()
    emb[emb > appearance_threshold] = 1.0
    emb[iou_dist > proximity_threshold] = 1.0
    return np.minimum(iou_fused, emb)
