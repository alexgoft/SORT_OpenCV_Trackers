import tensorflow as tf
import numpy as np

from config import DETECTOR_CONF, DETECTOR_CLASSES


class Detector:

    def __init__(self, path_to_ckpt):
        self.path_to_ckpt = path_to_ckpt
        self.detection_graph = tf.Graph()

        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = False  # dynamically grow the memory used on the GPU
        config.log_device_placement = True  # to log device placement (on which device the operation ran)
        self.sess = tf.Session(graph=self.detection_graph, config=config)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def predict(self, image):

        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)

        # Actual detection.
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})

        im_height, im_width, _ = image.shape

        boxes_out, scores_out, classes_out = [], [], []
        for i in range(boxes.shape[1]):
            score = scores[0, i]
            cls = classes[0, i]

            if score >= DETECTOR_CONF and cls in DETECTOR_CLASSES:
                box = (int(boxes[0, i, 0] * im_height),
                       int(boxes[0, i, 1] * im_width),
                       int(boxes[0, i, 2] * im_height),
                       int(boxes[0, i, 3] * im_width))

                boxes_out.append(box)
                scores_out.append(score)
                classes_out.append(cls)

        return boxes_out, scores_out, classes_out

