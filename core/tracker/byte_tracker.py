import numpy as np
import torch
from . import matching
from .basetrack import TrackState
from .kalman_filter import KalmanFilter
from .strack import STrack


class BYTETracker:
    """
    ByteTrack multi-object tracker.
    """
    def __init__(
            self,
            track_threshold: float = 0.5,
            track_buffer: int = 30,
            match_threshold: float = 0.8,
            frame_rate: int = 30,
            dup_iou_dist_threshold: float = 0.15
    ):
        """
        :param track_threshold: Score above which a detection is used for the first (high-confidence) association.
        :param track_buffer: Number of frames a lost track is kept before removal (scaled by the frame_rate).
        :param match_threshold: Maximum distance for a valid match in the first association.
        :param frame_rate: Video frame rate, used to size the lost-track buffer.
        :param dup_iou_dist_threshold: IoU distance (1 − IoU) under which overlapping tracks are merged as duplicates;
            the younger track is removed.
        """
        self.tracked_stracks: list[STrack] = []
        self.lost_stracks: list[STrack] = []
        self.removed_stracks: list[STrack] = []

        self.frame_id = 0
        self.track_threshold = track_threshold
        self.match_threshold = match_threshold
        self.det_thresh = track_threshold + 0.1
        self.buffer_size = int(frame_rate / 30.0 * track_buffer)
        self.max_time_lost = self.buffer_size
        self.dup_iou_dist_threshold = dup_iou_dist_threshold
        self.kalman_filter = KalmanFilter()

    def update(
        self,
        output_results: torch.Tensor,
        img_info: tuple,
        img_size: tuple,
        embeddings: np.ndarray | torch.Tensor | None = None,
    ) -> list[STrack]:
        """
        Advance the tracker by one frame and return the active tracks.

        :param output_results: Detections of shape (N, 5) [x1, y1, x2, y2, score]
            (a torch.Tensor from SCRFD.detect_tracking) or (N, 6) [..., score, cls].
        :param img_info: Original image size as (height, width).
        :param img_size: Model input size as (height, width).
        :param embeddings: Optional L2-normalized appearance features of shape (N, D),
            row-aligned with output_results. When provided, the first association fuses
            appearance with motion; otherwise it is IoU + score only.
        :return: List of activated tracks for the current frame.
        """
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        # Detections arrive as a torch tensor from SCRFD.detect_tracking; work in NumPy.
        if isinstance(output_results, torch.Tensor):
            output_results = output_results.cpu().numpy()
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]
        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes = bboxes / scale

        remain_ids = scores > self.track_threshold
        ids_low = scores > 0.1
        ids_high = scores < self.track_threshold

        ids_second = np.logical_and(ids_low, ids_high)
        dets_second = bboxes[ids_second]
        dets = bboxes[remain_ids]
        scores_keep = scores[remain_ids]
        scores_second = scores[ids_second]

        if embeddings is not None:
            feats_keep = embeddings[remain_ids]
        else:
            feats_keep = [None] * len(dets)

        if len(dets) > 0:
            # detection
            detections = [
                STrack(STrack.tlbr_to_tlwh(tlbr), s, feat=f)
                for (tlbr, s, f) in zip(dets, scores_keep, feats_keep)
            ]
        else:
            detections = []

        # Add newly detected tracklets to tracked_stracks
        unconfirmed = []
        tracked_stracks: list[STrack] = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        # step 2: First association, with high score detection boxes
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # predict the current location with KF
        STrack.multi_predict(strack_pool)
        ious_dist = matching.iou_distance(strack_pool, detections)
        if embeddings is not None:
            emb_dist = matching.embedding_distance(strack_pool, detections)
            dists = matching.fuse_appearance(emb_dist, ious_dist, detections)
        else:
            dists = matching.fuse_score(ious_dist, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, threshold=self.match_threshold)

        for itracked, idet in matches:
            itracked, idet = int(itracked), int(idet)
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # step 3: Second association, with low score detection boxes
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            # detection
            detections_second = [
                STrack(STrack.tlbr_to_tlwh(tlbr), s)
                for (tlbr, s) in zip(dets_second, scores_second)
            ]
        else:
            detections_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, _ = matching.linear_assignment(dists, threshold=0.5)
        for itracked, idet in matches:
            itracked, idet = int(itracked), int(idet)
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # deal with unconfirmed tracks, usually tracks with only one beginning frame
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, threshold=0.7)
        for itracked, idet in matches:
            itracked, idet = int(itracked), int(idet)
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        # step 4: Init new stracks
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)

        # step 5: Update state
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks, self.dup_iou_dist_threshold
        )
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        return output_stracks


def joint_stracks(tracks1: list[STrack], tracks2: list[STrack]) -> list[STrack]:
    """
    Merge two track lists, keeping each track id only once.

    :param tracks1: First track list.
    :param tracks2: Second track list.
    :return: Combined list with unique track ids.
    """
    exists = {}
    res = []
    for t in tracks1:
        exists[t.track_id] = 1
        res.append(t)
    for t in tracks2:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tracks1: list[STrack], tracks2: list[STrack]) -> list[STrack]:
    """
    Subtract one track list from another by track id.

    :param tracks1: Track list to subtract from.
    :param tracks2: Track list whose ids are removed from 'tracks1'.
    :return: Tracks in 'tracks1' whose ids are not in 'tracks2'.
    """
    stracks = {}
    for t in tracks1:
        stracks[t.track_id] = t
    for t in tracks2:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(tracks1: list[STrack], tracks2: list[STrack], iou_dist_threshold: float = 0.15) -> tuple[list[STrack], list[STrack]]:
    """
    Remove duplicate tracks between two lists based on IoU overlap.

    The track with the shorter lifetime in each overlapping pair is dropped.

    :param tracks1: First track list.
    :param tracks2: Second track list.
    :param iou_dist_threshold: IoU overlap threshold.
    :return: Filtered (tracks1, tracks2) with duplicates removed.
    """
    iou_dist = matching.iou_distance(tracks1, tracks2)
    pairs = np.where(iou_dist < iou_dist_threshold)
    dup_idx1, dup_idx2 = list(), list()
    for i1, i2 in zip(*pairs):
        i1, i2 = int(i1), int(i2)
        lifetime1 = tracks1[i1].frame_id - tracks1[i1].start_frame
        lifetime2 = tracks2[i2].frame_id - tracks2[i2].start_frame
        if lifetime1 > lifetime2:
            dup_idx2.append(i2)
        else:
            dup_idx1.append(i1)
    result1 = [t for i, t in enumerate(tracks1) if i not in dup_idx1]
    result2 = [t for i, t in enumerate(tracks2) if i not in dup_idx2]
    return result1, result2
