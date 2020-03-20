import cv2

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

#########################################################################################
# Trackers
#########################################################################################
OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "boosting": cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.TrackerTLD_create,
    "medianflow": cv2.TrackerMedianFlow_create,
    "mosse": cv2.TrackerMOSSE_create
}

#########################################################################################
# GENERAL
#########################################################################################
INPUT_VIDEO_PATH = "data/videos/man_walking.mp4"

SHOW_BBOXES = True
DISPLAY = True
WRITE = False