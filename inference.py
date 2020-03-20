#########################################################################################
#
#########################################################################################
import time
import cv2
import numpy as np
import tensorflow as tf

from object_detector import Detector
from tracker import Tracker
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
    chosen_tracker = OPENCV_OBJECT_TRACKERS[TRACKER_TYPE]
    mot_tracker = Tracker(DEFAULT_MAX_AGE,
                          DEFAULT_MIN_HITS,
                          DEFAULT_USE_TIME_SINCE_UPDATE,
                          DEFAULT_IOU_THRESHOLD,
                          chosen_tracker)

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

        frame_num += 1

        dets = []
        datas = []

        # --- GET DETECTIONS --- #
        boxes, scores, classes = detector.predict(image=image)
        for box, score, clas in zip(boxes, scores, classes):

            dets.append(box)
            datas.append([score, clas])

        # # --- GET PREDICTIONS --- #
        # tracks = mot_tracker.get_trackers_predictions(dets, datas, image_np)

        # --- SHOW IMAGES AND BOXES ---
        if DISPLAY:
            if SHOW_BBOXES:

                for det, data in zip(dets, datas):
                    xmin, ymin, xmax, ymax = [int(i) for i in det]
                    score = data[0]

                    # Display boxes.
                    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 5)
                    cv2.putText(image, str(score), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 5)

                # for track in tracks:
                #     xmin, ymin, xmax, ymax = [int(i) for i in track]
                #
                #     cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 255, 255), 2)

            cv2.imshow('MultiTracker', image)
            cv2.waitKey()

        # # --- Write to disk ---
        # if out is None:
        #     out_file_name = input_video_path.rsplit('.', 1)[0] + '_out.mp4'
        #     fourcc = cv2.VideoWriter_fourcc(*'XVID')
        #     out = cv2.VideoWriter(out_file_name, fourcc, 30.0, (width, height))
        #
        # out.write(image)

        # --- Print Stats ---
        # if frame_num % 50 == 0:
        #     print('Frame number: %f. FPS: %f' % (frame_num, 50.0 / (time.time() - frame_time_start)))
        #     frame_time_start = time.time()

    cap.release()
    out.release()


if __name__ == '__main__':
    main()
