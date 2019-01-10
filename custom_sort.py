#########################################################################################
#
#########################################################################################
from track import Track
import numpy as np
import utils


#########################################################################################
#
#########################################################################################
class Sort(object):
    _NUM_OF_COORDINATES = 4

    def __init__(self, max_age, min_hits, use_time_since_update, iou_threshold, tracker):
        """
    Sets key parameters for SORT
    """
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0
        self.use_time_since_update = use_time_since_update
        self.iou_threshold = iou_threshold
        self.tracker = tracker

    def update(self, dets, data, image):

        # ---- Initialize list of predictions with number of current trackers. ---- #
        curr_preds = np.zeros((len(self.trackers), self._NUM_OF_COORDINATES))
        failed_to_predict = []
        for p, _ in enumerate(curr_preds):
            success, pos = self.trackers[p].predict(image)
            if success:
                curr_preds[p] = np.array(utils.convert_xywh_to_bbox(pos))
            else:
                failed_to_predict.append(p)

        # Iterate from the end so indices of need-to-be-deleted elements will be preserved.
        for p in reversed(failed_to_predict):
            curr_preds = np.delete(curr_preds, p, axis=0)
            self.trackers.pop(p)

        matched, unmatched_dets, unmatched_trks = \
            utils.associate_detections_to_trackers(dets, curr_preds, self.iou_threshold)

        # ---- update matched trackers with assigned detections. ---- #
        failed_to_update = []
        for t, track in enumerate(self.trackers):
            if t not in unmatched_trks:
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                succ = track.update(image, utils.convert_bbox_to_xywh(dets[d, :][0]))
                if not succ:
                    failed_to_update.append(t)

        # Iterate from the end so indices of need-to-be-deleted elements will be preserved.
        for p in reversed(failed_to_update):
            self.trackers.pop(p)

        # ---- create and initialise new trackers for unmatched detections ---- #
        for i in unmatched_dets:
            new_tracker = Track(tracker=self.tracker)

            succ = new_tracker.update(image, utils.convert_bbox_to_xywh(dets[i]))
            if succ:
                self.trackers.append(new_tracker)

        # ---- Return predictions ---- #
        returned_preds = []  #
        i = len(self.trackers)
        for track in reversed(self.trackers):

            i -= 1

            success, pos = track.get_state(image)
            if not success or track.time_since_update > self.max_age:
                self.trackers.pop(i)
                continue

            # if (track.time_since_update < self.use_time_since_update) and \
            #         (track.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
            if (track.time_since_update < self.use_time_since_update) and \
                    (track.hits >= self.min_hits or self.frame_count <= self.min_hits):

                returned_preds.append(utils.convert_xywh_to_bbox(pos))

        if len(returned_preds) > 0:
            return returned_preds
        return np.empty((0, 5))
