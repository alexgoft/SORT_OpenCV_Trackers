import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment


#########################################################################################
#
#########################################################################################
# def convert_bbox_to_z(bbox):
#     """
#   Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
#     [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
#     the aspect ratio
#   """
#     w = bbox[2] - bbox[0]
#     h = bbox[3] - bbox[1]
#     x = bbox[0] + w / 2.
#     y = bbox[1] + h / 2.
#     s = w * h  # scale is just area
#     r = w / float(h)
#     return np.array([x, y, s, r]).reshape((4, 1))
#
#
# def convert_x_to_bbox(x, score=None):
#     """
#   Takes a bounding box in the form [x,y,s,r] and returns it in the form
#     [x1,y1,x2,x2] where x1,y1 is the top left and x2,y2 is the bottom right
#   """
#     w = np.sqrt(x[2] * x[3])
#     h = x[2] / w
#     if score == None:
#         return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
#     else:
#         return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))


def convert_xywh_to_bbox(xywh):
    return xywh[0], xywh[1], xywh[0] + xywh[2], xywh[1] + xywh[3]


def convert_bbox_to_xywh(det):
    return det[0], det[1], det[2] - det[0], det[3] - det[1]


#########################################################################################
#
#########################################################################################
def iou(bb_test, bb_gt):
    """
  Computes IUO between two bboxes in the form [x1,y1,x2,y2]
  """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
              + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
    return o


def associate_detections_to_trackers(detections, preds, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if len(preds) == 0:
        return [], np.arange(len(detections)), []

    # ------ FIND MATCHES --------- #
    iou_matrix = np.zeros((len(detections), len(preds)), dtype=np.float32)

    for d, det in enumerate(detections):
        for t, trk in enumerate(preds):
            iou_matrix[d, t] = iou(det, trk)
    matched_indices = linear_assignment(-iou_matrix)

    # ------- FIND UNMATCHED DETECTIONS -------- #
    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)

    # ------- FIND MATCHED DETECTIONS -------- #
    unmatched_trackers = []
    for t, trk in enumerate(preds):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # -------filter out matched with low IOU-------- #
    matches = []
    for match in matched_indices:
        if iou_matrix[match[0], match[1]] < iou_threshold:
            unmatched_detections.append(match[0])
            unmatched_trackers.append(match[1])
        else:
            matches.append(match)

    return np.array(matches), np.array(unmatched_detections), np.array(unmatched_trackers)
