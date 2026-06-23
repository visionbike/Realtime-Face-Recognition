from collections import deque
import numpy as np
from .basetrack import BaseTrack, TrackState
from .kalman_filter import KalmanFilter


class STrack(BaseTrack):
    """
    Single-object track holding Kalman state for one detection.

    Wraps a Kalman-filtered motion state for one detection and optionally, an EMA-smoothed appearance (ReID) feature.
    The bounding box is stored internally in 'tlwh' format (top-left x, top-left y, width, height).
    """

    shared_kalman = KalmanFilter()      # class-level Kalman filter shared by multi_predict for vectorized prediction
                                        # across many tracks at once.

    def __init__(self, tlwh: np.ndarray, score: float, feat: np.ndarray | None = None, feat_history: int = 50):
        """

        :param tlwh: Initial bounding box as (top-left x, top-left y, width, height).
        :param score: Detection confidence score.
        :param feat: Optional L2-normalized appearance (ReID) feature of shape (D,).
        :param feat_history: Number of recent features to retain for this track.
        """
        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float64)
        self.kalman_filter: KalmanFilter | None = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        # appearance / ReID state — always declared so the type is known on every path.
        self.curr_feat: np.ndarray | None = None
        self.smooth_feat: np.ndarray | None = None
        self.features: deque[np.ndarray] = deque(maxlen=feat_history)
        self.alpha = 0.9
        if feat is not None:
            self.update_features(feat)

    def update_features(self, feat: np.ndarray):
        """
        Update the current and EMA-smoothed appearance features.

        :param feat: New L2-normalized appearance feature of shape (D,).
        """
        feat = feat / np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            smooth_feat = feat
        else:
            smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        smooth_feat = smooth_feat / np.linalg.norm(smooth_feat)
        self.smooth_feat = smooth_feat
        self.features.append(feat)

    def predict(self):
        """
        Advance the track state one step using the Kalman motion model.
        """
        assert self.kalman_filter is not None, "predict() called before activate()."
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks: list["STrack"]):
        """
        Vectorized Kalman prediction for a list of tracks (updated in place).

        :param stracks: Tracks to advance one step.
        """
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter: KalmanFilter, frame_id: int):
        """
        Start a new tracklet.

        :param kalman_filter: Kalman filter instance to own this track's state.
        :param frame_id: Current frame index.
        """
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track: "STrack", frame_id: int, new_id: bool = False):
        """
        Reactivate a lost track from a new matched detection.

        :param new_track: Newly matched detection track.
        :param frame_id: Current frame index.
        :param new_id: If True, assign a new track id.
        """
        assert self.kalman_filter is not None, "re_activate() called before activate()."
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean,
            self.covariance,
            self.tlwh_to_xyah(new_track.tlwh)
        )
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track: "STrack", frame_id: int):
        """
        Update a matched track with a new detection.

        :param new_track: Newly matched detection track.
        :param frame_id: Current frame index.
        """
        assert self.kalman_filter is not None, "update() called before activate()."
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean,
            self.covariance,
            self.tlwh_to_xyah(new_tlwh)
        )
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

    @property
    def tlwh(self) -> np.ndarray:
        """
        Current position as (top-left x, top-left y, width, height).
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self) -> np.ndarray:
        """
        Current position as (top-left x, top-left y, bottom-right x, bottom-right y).
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh: np.ndarray) -> np.ndarray:
        """
        Convert (top-left x, top-left y, width, height) to (center x, center y, aspect ratio, height).

        :param tlwh: Box as (top-left x, top-left y, width, height).
        :return: Box as (center x, center y, aspect ratio, height).
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self) -> np.ndarray:
        """
        Return this track's box as (center x, center y, aspect ratio, height).
        """
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr: np.ndarray) -> np.ndarray:
        """
        Convert (top-left x, top-left y, bottom-right x, bottom-right y) to (top-left x, top-left y, width, height).

        :param tlbr: Box as (top-left x, top-left y, bottom-right x, bottom-right y).
        :return: Box as (top-left x, top-left y, width, height).
        """
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh: np.ndarray) -> np.ndarray:
        """
        Convert (top-left x, top-left y, width, height) to (top-left x, top-left y, bottom-right x, bottom-right y).

        :param tlwh: Box as (top-left x, top-left y, width, height).
        :return: Box as (top-left x, top-left y, bottom-right x, bottom-right y).
        """
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return "OT_{}_({}-{})".format(self.track_id, self.start_frame, self.end_frame)
