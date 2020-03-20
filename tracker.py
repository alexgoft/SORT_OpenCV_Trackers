import numpy

from sort import Sort


class Tracker(object):
    def __init__(self, max_age, min_hits, use_time_since_update, iou_threshold, tracker):
        self._mot_tracker = Sort(max_age=max_age, min_hits=min_hits,
                                 use_time_since_update=use_time_since_update,
                                 iou_threshold=iou_threshold, tracker=tracker)

    def get_trackers_predictions(self, seq_dets, seq_data, image):

        seq_dets = numpy.array(seq_dets)
        seq_data = numpy.array(seq_data)

        tracked_targets = self._mot_tracker.update(seq_dets, seq_data, image)

        # tracked_targets.extend(self._mot_tracker.update(seq_dets, seq_data, image_np))
        return tracked_targets
