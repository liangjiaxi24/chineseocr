from text.keras_yolo3 import yolo_text
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2


p = '../test/idcard-demo.jpeg'
img = cv2.imread(p)
h, w = img.shape[:2]



keras_anchors = '8,11, 8,16, 8,23, 8,33, 8,48, 8,97, 8,139, 8,198, 8,283'
class_names = ['none', 'text']
kerasTextModel = "../models/text.h5"
scale = 600
maxScale = 900

anchors = [float(x) for x in keras_anchors.split(',')]
anchors = np.array(anchors).reshape(-1, 2)

num_anchors = len(anchors)
num_classes = len(class_names)

textModel = yolo_text(num_classes, anchors)
textModel.load_weights(kerasTextModel)


boxes, scores = textModel(img, 608, 2048)
print(boxes)
print(scores)
