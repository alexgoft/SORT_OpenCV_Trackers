#########################################################################################
#
#########################################################################################
import time
import cv2

from object_detector import Detector
from sort import Sort
from config import *


#########################################################################################
#
#########################################################################################

def main():

    # ----------------------------------------- #
    # INITIALIZE DETECTOR
    # ----------------------------------------- #
    detector = Detector(path_to_ckpt=DETECTOR_GRAPH_PATH)

    # ----------------------------------------- #
    # INITIALIZE TRACKER
    # ----------------------------------------- #
    mot_tracker = Sort(max_age=DEFAULT_MAX_AGE,
                       min_hits=DEFAULT_MIN_HITS,
                       use_time_since_update=DEFAULT_USE_TIME_SINCE_UPDATE,
                       iou_threshold=DEFAULT_IOU_THRESHOLD,
                       tracker_type=TRACKER_TYPE)

    # ----------------------------------------- #
    # INITIALIZE STREAM
    # ----------------------------------------- #
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    out = None

    # ----------------------------------------- #
    frame_num = 0

    while cap.isOpened():
        ret, image = cap.read()
        if ret == 0:
            break

        frame_time_start = time.time()

        dets = []
        datas = []

        # --- GET DETECTIONS --- #
        boxes, scores, classes = detector.predict(image=image)
        for box, score, clas in zip(boxes, scores, classes):
            dets.append(box)
            datas.append([score, clas])

        # --- GET PREDICTIONS --- #
        tracks, tracks_ids = mot_tracker.update_and_get_tracks(dets, image)

        # --- SHOW IMAGES AND BOXES --- #
        if DISPLAY:
            if SHOW_BBOXES:

                for det, data in zip(dets, datas):
                    xmin, ymin, xmax, ymax = [int(i) for i in det]

                    # Display boxes.
                    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 255, 255), 5)

                for id, track in zip(tracks_ids, tracks):
                    xmin, ymin, xmax, ymax = [int(i) for i in track]

                    cv2.rectangle(image, pt1=(xmin, ymin), pt2=(xmax, ymax), color=TRACKER_COLORS[id], thickness=2)

            cv2.imshow('MultiTracker', image)
            cv2.waitKey()

        # --- Write to disk --- #
        if WRITE:
            if out is None:
                out_file_name = INPUT_VIDEO_PATH.rsplit('.', 1)[0] + '_out.mp4'
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(out_file_name, fourcc, 30.0, (image.shape[1], image.shape[0]))

            out.write(image)

        # --- Print Stats ---
        print('FRAME NUMBER: %d. FPS: %f' % (frame_num, 1 / (time.time() - frame_time_start)))

        frame_num += 1

    cap.release()
    if out is not None:
        out.release()


if __name__ == '__main__':
    main()
