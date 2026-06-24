from collections import OrderedDict
import numpy as np


class TrackState:
    """
    Enumerate of the lifecycle states a track can be in.
    """

    New = 0         # created from a detection, not yet confirmed.
    Tracked = 1     # actively matched and updated in the current frame.
    Lost = 2        # temporarily unmatched; kept alive in the lost buffer.
    Removed = 3     # discarded; no longer tracked.


class BaseTrack:
    """
    Abstract base class for a single object track.
    """

    _count = 0                  # class-level counter backing next_id for globally unique ids.

    track_id = 0                # unique id assigned when the track is activated.
    is_activated = False        # whether the track has been confirmed.
    state = TrackState.New      # current lifecycle state.

    history = OrderedDict()     # ordered record of past states/observations.
    features = []               # stored appearance (ReID) features for this track.
    curr_feature = None         # most recent appearance feature.
    score = 0                   # detection confidence of the latest observation.
    start_frame = 0             # frame index where the track began.
    frame_id = 0                # most recent frame index in which the track updated.
    time_since_update = 0       # frames elapsed since the last update.

    # multi-camera
    location = (np.inf, np.inf) # multi-camera world location, `(inf, inf)` until set.

    @property
    def end_frame(self) -> int:
        """
        Last frame index in which this track was updated.
        """
        return self.frame_id

    @staticmethod
    def next_id() -> int:
        """
        Return a new globally unique track id.
        """
        BaseTrack._count += 1
        return BaseTrack._count

    def activate(self, *args):
        """
        Start a new tracklet. Must be implemented by subclasses.
        """
        raise NotImplementedError

    def predict(self):
        """
        Advance the track state one step. Must be implemented by subclasses.
        """
        raise NotImplementedError

    def update(self, *args, **kwargs):
        """
        Update a matched track. Must be implemented by subclasses.
        """
        raise NotImplementedError

    def mark_lost(self):
        """
        Mark this track as lost.
        """
        self.state = TrackState.Lost

    def mark_removed(self):
        """
        Mark this track as removed.
        """
        self.state = TrackState.Removed
