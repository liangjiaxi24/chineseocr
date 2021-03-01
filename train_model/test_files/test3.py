import os
os.environ["CUDA_VISIBLE_DEVICES"] = ''
import cv2
from text.keras_detect import text_detect, textModel


def detect_box(img, scale=600, maxScale=900):
    """
    detect text angle in [0,90,180,270]
    @@img:np.array
    """
    boxes, scores = textModel(img, scale, maxScale)
    return boxes, scores


p = '../test/idcard-demo.jpeg'
img = cv2.imread(p)
detect_box(img)


