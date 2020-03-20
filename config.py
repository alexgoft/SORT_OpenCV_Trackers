import cv2

#########################################################################################
# DETECTOR
#########################################################################################
DETECTOR_GRAPH_PATH = 'data/videos/people_walking.mp4'
DETECTOR_CONF = 0.8
DETECTOR_CLASSES = {1: "PERSON"}

#########################################################################################
# Tracker Arguments
#########################################################################################
DEFAULT_MAX_AGE = 40
DEFAULT_MIN_HITS = 1
DEFAULT_USE_TIME_SINCE_UPDATE = 200
DEFAULT_IOU_THRESHOLD = 0.5
DEFAULT_MIN_SCORE = 0.2
TRACKER_TYPE = 'kalman'

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
INPUT_VIDEO_PATH = ""
PLOT_TRACKS_COLOR = "Violet"  # None to disable
PLOT_RAW_DETECTIONS_COLOR = None  # None to disable
DETECTION_BBOX_CENTER_DOT_COLOR = None  # None to disable
BBOX_FRAME_RATIO_THRESHOLD = 0.2
SCORE_THRESHOLD = 0.5
ENLARGE_BBOX_PERCANTAGE = 0.03
SHOW_BBOXES = True
BLUR_BBOXES = True
DISPLAY = True