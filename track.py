class Track(object):

    def __init__(self, tracker):
        self._tracker_constructor = tracker
        self._tracker = None

        self.time_since_update = 0
        self.hits = 0

        self.hit_streak = 0

    def update(self, image, bbox):
        """
            Cant reinitialize after creation. So Best practice is to create a new object.

            https://stackoverflow.com/questions/49755892/updating-an-opencv-tracker-with-a-bounding-box-in-python
            https://stackoverflow.com/questions/31432815/opencv-3-tracker-wont-work-after-reinitialization


            box = utils.convert_bbox_to_xywh(dets[i])
            tracker = self.tracker()
            x = tracker.init(image_np, box) # True
            y = tracker.init(image_np, box) # False
        """

        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

        self._tracker = self._tracker_constructor()
        return self._tracker.init(image, bbox)

    def get_state(self, image):
        return self._tracker.update(image)

    def predict(self, image):
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        return self.get_state(image)
