from crnn.keys import alphabetChinese
from crnn.network_torch import CRNN
from PIL import Image
import numpy as np
import cv2


ocr_path = '/root/ocr_project_dir/chineseocr/models/ocr-lstm.pth'
alphabet = alphabetChinese
nclass = len(alphabet) + 1

crnn_model = CRNN(32, 1, nclass, 256, leakyRelu=False, lstmFlag=True, GPU=False, alphabet=alphabet)
crnn_model.load_weights(ocr_path)


def solve(box):
    """
    绕 cx,cy点 w,h 旋转 angle 的坐标
    x = cx-w/2
    y = cy-h/2
    x1-cx = -w/2*cos(angle) +h/2*sin(angle)
    y1 -cy= -w/2*sin(angle) -h/2*cos(angle)

    h(x1-cx) = -wh/2*cos(angle) +hh/2*sin(angle)
    w(y1 -cy)= -ww/2*sin(angle) -hw/2*cos(angle)
    (hh+ww)/2sin(angle) = h(x1-cx)-w(y1 -cy)

    """
    x1, y1, x2, y2, x3, y3, x4, y4 = box[:8]
    cx = (x1 + x3 + x2 + x4) / 4.0
    cy = (y1 + y3 + y4 + y2) / 4.0
    w = (np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) + np.sqrt((x3 - x4) ** 2 + (y3 - y4) ** 2)) / 2
    h = (np.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2) + np.sqrt((x1 - x4) ** 2 + (y1 - y4) ** 2)) / 2
    # x = cx-w/2
    # y = cy-h/2

    sinA = (h * (x1 - cx) - w * (y1 - cy)) * 1.0 / (h * h + w * w) * 2
    if abs(sinA) > 1:
        angle = None
    else:
        angle = np.arcsin(sinA)
    return angle, w, h, cx, cy


def rotate_cut_img(im, box, leftAdjustAlph=0.0, rightAdjustAlph=0.0):
    angle, w, h, cx, cy = solve(box)
    degree_ = angle * 180.0 / np.pi

    box = (max(1, cx - w / 2 - leftAdjustAlph * (w / 2))  ##xmin
           , cy - h / 2,  ##ymin
           min(cx + w / 2 + rightAdjustAlph * (w / 2), im.size[0] - 1)  ##xmax
           , cy + h / 2)  ##ymax
    newW = box[2] - box[0]
    newH = box[3] - box[1]
    tmpImg = im.rotate(degree_, center=(cx, cy)).crop(box)
    box = {'cx': cx, 'cy': cy, 'w': newW, 'h': newH, 'degree': degree_, }
    return tmpImg, box


def ocr_batch(img, boxes, leftAdjustAlph=0.0, rightAdjustAlph=0.0):
    """
    batch for ocr
    """
    im = Image.fromarray(img)
    newBoxes = []
    for index, box in enumerate(boxes):
        partImg, box = rotate_cut_img(im, box, leftAdjustAlph, rightAdjustAlph)
        box['img'] = partImg.convert('L')
        newBoxes.append(box)

        partImg.save("./train_model/part_{}.png".format(index))

    res = crnn_model.predict_job(newBoxes)
    return res




# boxes = [np.array([47, 158, 341, 159, 340, 250,  47, 249]),
#          np.array([567, 159, 739, 157, 740, 249, 568, 251])]
# # boxes = [np.array([0, 0, 843, 0, 0, 420, 843, 420])]
# img = cv2.imread("./train_model/2.jpg")
# res = ocr_batch(img, boxes, 0.01, 0.01)
# print(res)


img = Image.open("./train_model/2.png")
img = img.convert('L')
raw = crnn_model.predict(img)
print(raw)
