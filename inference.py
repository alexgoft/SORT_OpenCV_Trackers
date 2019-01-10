#########################################################################################
#
#########################################################################################
import time
from random import randint
import PIL.Image as Image
import cv2
import numpy as np
from tracker import Tracker
import tensorflow as tf
import blur
from tensorflow_face_detection.utils import visualization_utils_color as vis_util
from tensorflow_face_detection.utils.tf_utils import load_frozen_graph, get_detection_tensors

#########################################################################################
#
#########################################################################################
flags = tf.app.flags
flags.DEFINE_string('label_map_path', '', 'Path to label map file.')
flags.DEFINE_string('frozen_graph_path', '', 'Path to frozen graph file.')
flags.DEFINE_string('input_video_path', '', 'Path to input video file.')
flags.DEFINE_string('start_frame', '0', 'Path to input video file.')
flags.DEFINE_string('blur_type', 'pool', 'Blur type (gaussian, mode, pool).')
flags.DEFINE_string('tracker', 'csrt',
                    'Tracker type can be - '
                    '\'BOOSTING\', \'MIL\', \'KCF\', \'TLD\', \'MEDIANFLOW\', \'GOTURN\', \'MOSSE\', \'CSRT\'')

FLAGS = flags.FLAGS

#########################################################################################
# Tracker Arguments
#########################################################################################
DEFAULT_MAX_AGE = 40
DEFAULT_MIN_HITS = 1
DEFAULT_USE_TIME_SINCE_UPDATE = 200
DEFAULT_IOU_THRESHOLD = 0.5
DEFAULT_MIN_SCORE = 0.2

#########################################################################################
#
#########################################################################################
PLOT_TRACKS_COLOR = "Violet"  # None to disable
PLOT_RAW_DETECTIONS_COLOR = None  # None to disable
DETECTION_BBOX_CENTER_DOT_COLOR = None  # None to disable
BBOX_FRAME_RATIO_THRESHOLD = 0.2
SCORE_THRESHOLD = 0.5
ENLARGE_BBOX_PERCANTAGE = 0.03
SHOW_BBOXES = True
BLUR_BBOXES = True
DISPLAY = True

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
#
#########################################################################################
def enlarge_bbox(height, width, xmax, xmin, ymax, ymin):
    # Enlarge BBOX
    xmin = max(0, int(xmin * (1 - ENLARGE_BBOX_PERCANTAGE)))
    ymin = max(0, int(ymin * (1 - ENLARGE_BBOX_PERCANTAGE)))
    xmax = min(width, int(xmax * (1 + ENLARGE_BBOX_PERCANTAGE)))
    ymax = min(height, int(ymax * (1 + ENLARGE_BBOX_PERCANTAGE)))
    det = [xmin, ymin, xmax, ymax]
    return det


def main():
    # ----------------------------------------- #
    frozen_graph_path = FLAGS.frozen_graph_path
    input_video_path = FLAGS.input_video_path
    blur_type = FLAGS.blur_type
    tracker = FLAGS.tracker
    start_frame = FLAGS.start_frame

    # ----------------------------------------- #
    chosen_tracker = OPENCV_OBJECT_TRACKERS[tracker]
    mot_tracker = Tracker(DEFAULT_MAX_AGE,
                          DEFAULT_MIN_HITS,
                          DEFAULT_USE_TIME_SINCE_UPDATE,
                          DEFAULT_IOU_THRESHOLD,
                          chosen_tracker)
    cap = cv2.VideoCapture(input_video_path)
    out = None

    # ----------------------------------------- #
    frame_num = 0
    frames_per_time = 0

    detection_graph = load_frozen_graph(frozen_graph_path)
    with detection_graph.as_default():

        # config = tf.ConfigProto(device_count={'GPU': 0})
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(graph=detection_graph, config=config) as sess:
            boxes_tensor, classes_tensor, image_tensor, scores_tensor = get_detection_tensors(detection_graph)

            frame_time_start = time.time()

            while cap.isOpened():
                ret, image = cap.read()
                if ret == 0:
                    break

                frame_num += 1
                frames_per_time += 1

                if frame_num < int(start_frame):
                    cap.read()

                    continue

                dets = []
                datas = []

                # --- GET DETECTIONS --- #
                image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_np_expanded = np.expand_dims(image_np, axis=0)

                height, width = image_np.shape[0], image_np.shape[1]

                (boxes, scores, classes) = sess.run([boxes_tensor, scores_tensor, classes_tensor],
                                                    feed_dict={image_tensor: image_np_expanded})
                boxes, scores, classes = np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes).astype(np.int32)
                for box, score, clas in zip(boxes, scores, classes):

                    if score > SCORE_THRESHOLD and clas in [1]:
                        # if score > SCORE_THRESHOLD and clas in [1, 2, 4]:
                        ymin, xmin, ymax, xmax = [int(i * size) for i, size in zip(box, [height, width, height, width])]

                        det = enlarge_bbox(height, width, xmax, xmin, ymax, ymin)
                        data = [score, clas]

                        dets.append(det)
                        datas.append(data)

                # --- GET PREDICTIONS --- #
                tracks = mot_tracker.get_trackers_predictions(dets, datas, image_np)

                # --- BLURING ---
                if BLUR_BBOXES:
                    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
                    for track in tracks:
                        blur.draw_blurred_box_on_image(image, image_pil, width, height, track, blur_type)

                # --- SHOW IMAGES AND BOXES ---
                if DISPLAY:
                    if SHOW_BBOXES:

                        for det, data in zip(dets, datas):
                            xmin, ymin, xmax, ymax = [int(i) for i in det]
                            score = data[0]

                            # Display boxes.
                            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 5)
                            cv2.putText(image, str(score), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 5)

                        for track in tracks:
                            xmin, ymin, xmax, ymax = [int(i) for i in track]

                            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 255, 255), 2)

                    cv2.imshow('MultiTracker', image)

                    # quit on ESC button
                    if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
                        break

                # # --- Write to disk ---
                # if out is None:
                #     out_file_name = input_video_path.rsplit('.', 1)[0] + '_out.mp4'
                #     fourcc = cv2.VideoWriter_fourcc(*'XVID')
                #     out = cv2.VideoWriter(out_file_name, fourcc, 30.0, (width, height))
                #
                # out.write(image)

                # --- Print Stats ---
                if frame_num % 50 == 0:
                    print('Frame number: %f. FPS: %f' % (frame_num, 50.0 / (time.time() - frame_time_start)))
                    frame_time_start = time.time()

            cap.release()
            out.release()


if __name__ == '__main__':
    main()
