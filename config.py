import numpy as np

#########################################################################################
# DETECTOR
#########################################################################################
DETECTOR_GRAPH_PATH = 'data/checkpoints/faster_rcnn_inception_v2_coco/frozen_inference_graph.pb'
DETECTOR_CONF = 0.95
DETECTOR_CLASSES = {1: "PERSON"}

#########################################################################################
# Tracker Arguments
#########################################################################################
DEFAULT_MAX_AGE = 40
DEFAULT_MIN_HITS = 1
DEFAULT_USE_TIME_SINCE_UPDATE = 200
DEFAULT_IOU_THRESHOLD = 0.5
DEFAULT_MIN_SCORE = 0.2
TRACKER_TYPE = 'medianflow'
TRACKER_COLORS = [tuple([int(x) for x in np.random.choice(range(256), size=3)]) for _ in range(50)]


#########################################################################################
# GENERAL
#########################################################################################
INPUT_VIDEO_PATH = "/hdd/SORT_OpenCV_Trackers/data/videos/people.mp4"

SHOW_BBOXES = True
DISPLAY = True
WRITE = False