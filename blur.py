import numpy as np
import PIL.Image as Image

#########################################################################################
#
#########################################################################################
GAUSSIAN_BLUR_KEY = "gaussian"
MODE_BLUR_KEY = "mode"
POOLING_BLUR_KEY = "pool"


#########################################################################################
#
#########################################################################################
class ProportionalPoolingBlur(object):
    """
    Performs pooling blurring with factor proportional to bbox size
    """
    DEFAULT_SLOPE = 50
    DEFAULT_INTERCEPT = 2

    # DEFAULT_SLOPE = 50
    # DEFAULT_INTERCEPT = 10

    def __init__(self, bbox, slope=None, intercept=None):
        """
        :param bbox: (xmin, xmax, ymin, ymax) tuple of bounding box
        :param slope: linear slope for factor calculation
        :param intercept: linear intercept for factor calculation
        """
        slope = slope if slope is not None else self.DEFAULT_SLOPE
        intercept = intercept if intercept is not None else self.DEFAULT_INTERCEPT
        self._factor = calculate_linear_factor_from_bbox(bbox, slope, intercept)

    def filter(self, image):
        """
        conform with PIL's Filter classes
        """
        image_height = image.size[0]
        image_width = image.size[1]

        pooled_image = image.resize((max(1, image_height / self._factor), max(1, image_width / self._factor)),
                                    Image.NEAREST)
        pooled_image = pooled_image.resize((image_height, image_width), Image.NEAREST)

        return pooled_image


def calculate_linear_factor_from_bbox(bbox, slope, intercept):
    xmin, xmax, ymin, ymax = bbox
    bbox_width = xmax - xmin
    bbox_height = ymax - ymin
    bbox_scale = np.mean([bbox_width, bbox_height])
    return int(intercept + slope * bbox_scale)


def draw_blurred_box_on_image(image, pil_image, image_width, image_height, box, blur_type='pool'):
    x1, y1, x2, y2 = [int(b) for b in box]

    (left, right, top, bottom) = (x1, x2, y1, y2)

    cropped_img = pil_image.crop((left, top, right, bottom))

    box_coordinates = (x1 / float(image_width), x2 / float(image_width),
                       y1 / float(image_height), y2 / float(image_height))

    if POOLING_BLUR_KEY == blur_type:
        blurred_patch = cropped_img.filter(ProportionalPoolingBlur(tuple(box_coordinates)))
    # elif GAUSSIAN_BLUR_KEY == blur_type:
    #     blurred_patch = cropped_img.filter(ImageFilter.GaussianBlur(20))
    # elif MODE_BLUR_KEY == blur_type:
    #     blurred_patch = cropped_img.filter(ProportionalModeFilter(tuple(box_coordinates)))
    else:
        blurred_patch = cropped_img

    blurred_patch = np.array(blurred_patch)
    try:
        image[y1:y2, x1:x2, :] = blurred_patch
    except:
        pass
