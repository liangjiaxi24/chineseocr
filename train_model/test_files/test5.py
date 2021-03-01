import cv2
import numpy as np
from crnn.keys import alphabetChinese
from main import TextOcrModel
import time
from PIL import Image
from config import *
from crnn.network_torch import CRNN
from text.keras_detect import text_detect
from text.opencv_dnn_detect import angle_detect


if yoloTextFlag == 'keras' or AngleModelFlag == 'tf' or ocrFlag == 'keras':
    os.environ["CUDA_VISIBLE_DEVICES"] = ''


alphabet = alphabetChinese
nclass = len(alphabet) + 1
crnn = CRNN(32, 1, nclass, 256, leakyRelu=False, lstmFlag=LSTMFLAG, GPU=GPU, alphabet=alphabet)
ocr = crnn.predict_job

model = TextOcrModel(ocr, text_detect, angle_detect)
from apphelper.image import xy_rotate_box,box_rotate,solve


p = '../test/idcard-demo.jpeg'
img = cv2.imread(p)

h, w = img.shape[:2]
timeTake = time.time()
scale = 608
maxScale = 2048


def plot_boxes(img, angle, result, color=(0, 0, 0)):
    tmp = np.array(img)
    c = color
    h, w = img.shape[:2]
    thick = int((h + w) / 300)
    i = 0
    if angle in [90, 270]:
        imgW, imgH = img.shape[:2]

    else:
        imgH, imgW = img.shape[:2]

    for line in result:
        cx = line['cx']
        cy = line['cy']
        degree = line['degree']
        w = line['w']
        h = line['h']

        x1, y1, x2, y2, x3, y3, x4, y4 = xy_rotate_box(cx, cy, w, h, degree / 180 * np.pi)

        x1, y1, x2, y2, x3, y3, x4, y4 = box_rotate([x1, y1, x2, y2, x3, y3, x4, y4], angle=(360 - angle) % 360,
                                                    imgH=imgH, imgW=imgW)
        cx = np.mean([x1, x2, x3, x4])
        cy = np.mean([y1, y2, y3, y4])
        cv2.line(tmp, (int(x1), int(y1)), (int(x2), int(y2)), c, 1)
        cv2.line(tmp, (int(x2), int(y2)), (int(x3), int(y3)), c, 1)
        cv2.line(tmp, (int(x3), int(y3)), (int(x4), int(y4)), c, 1)
        cv2.line(tmp, (int(x4), int(y4)), (int(x1), int(y1)), c, 1)
        mess = str(i)
        cv2.putText(tmp, mess, (int(cx), int(cy)), 0, 1e-3 * h, c, thick // 2)
        i += 1
    return Image.fromarray(tmp).convert('RGB')


def plot_rboxes(img, boxes, color=(0, 0, 0)):
    tmp = np.array(img)
    c = color
    h, w = img.shape[:2]
    thick = int((h + w) / 300)
    i = 0

    for box in boxes:
        x1, y1, x2, y2, x3, y3, x4, y4 = box

        cx = np.mean([x1, x2, x3, x4])
        cy = np.mean([y1, y2, y3, y4])
        cv2.line(tmp, (int(x1), int(y1)), (int(x2), int(y2)), c, 1)
        cv2.line(tmp, (int(x2), int(y2)), (int(x3), int(y3)), c, 1)
        cv2.line(tmp, (int(x3), int(y3)), (int(x4), int(y4)), c, 1)
        cv2.line(tmp, (int(x4), int(y4)), (int(x1), int(y1)), c, 1)
        mess = str(i)
        cv2.putText(tmp, mess, (int(cx), int(cy)), 0, 1e-3 * h, c, thick // 2)
        i += 1
    return Image.fromarray(tmp).convert('RGB')

result, angle = model.model(img,
                            detectAngle=True,  ##是否进行文字方向检测
                            scale=scale,
                            maxScale=maxScale,
                            MAX_HORIZONTAL_GAP=80,  ##字符之间的最大间隔，用于文本行的合并
                            MIN_V_OVERLAPS=0.6,
                            MIN_SIZE_SIM=0.6,
                            TEXT_PROPOSALS_MIN_SCORE=0.1,
                            TEXT_PROPOSALS_NMS_THRESH=0.7,
                            TEXT_LINE_NMS_THRESH=0.9,  ##文本行之间测iou值
                            LINE_MIN_SCORE=0.1,
                            leftAdjustAlph=0,  ##对检测的文本行进行向左延伸
                            rightAdjustAlph=0.1,  ##对检测的文本行进行向右延伸
                            )

timeTake = time.time() - timeTake

print('It take:{}s'.format(timeTake))
for line in result:
    print(line['text'])
plot_boxes(img, angle, result, color=(0, 0, 0))