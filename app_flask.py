import json
import time
import uuid
import numpy as np
from PIL import Image
from config import *
from flask import Flask, request
from application import trainTicket, idcard
from apphelper.image import union_rbox, adjust_box_to_origin, base64_to_PIL
from app_main import billList, filelock, crnn, scale, maxScale, model


app = Flask(__name__)


@app.route("/ocr/", methods=["POST", "GET"])
def run_ocr():
    if request.method == "GET":
        post = {}
        ## 请求地址
        post['postName'] = 'ocr'
        post['height'] = 1000
        post['H'] = 1000
        post['width'] = 600
        post['W'] = 600
        post['billList'] = billList
        return json.dumps(post, ensure_ascii=False)

    elif request.method == "POST":
        t = time.time()
        data = request.get_data(as_text=True)
        if data:
            data = json.loads(data)
        else:
            return json.dumps({'res': "no inputs", 'timeTake': 0}, ensure_ascii=False)

        uidJob = uuid.uuid1().__str__()
        billModel = data.get('billModel', '')
        textAngle = data.get('textAngle', False)  ##文字检测
        textLine = data.get('textLine', False)  ##只进行单行识别

        imgString = data['imgString'].encode().split(b';base64,')[-1]
        img = base64_to_PIL(imgString)
        if img is not None:
            img = np.array(img)

        H, W = img.shape[:2]

        while time.time() - t <= TIMEOUT:
            if os.path.exists(filelock):
                continue
            else:
                with open(filelock, 'w') as f:
                    f.write(uidJob)

                if textLine:
                    # 单行识别
                    partImg = Image.fromarray(img)
                    text = crnn.predict(partImg.convert('L'))
                    res = [{'text': text, 'name': '0', 'box': [0, 0, W, 0, W, H, 0, H]}]
                    os.remove(filelock)
                    break

                else:
                    detectAngle = textAngle
                    result, angle = model.model(img,
                                                scale=scale,
                                                maxScale=maxScale,
                                                # 是否进行文字方向检测，通过web传参控制
                                                detectAngle=detectAngle,
                                                # 字符之间的最大间隔，用于文本行的合并
                                                MAX_HORIZONTAL_GAP=100,
                                                MIN_V_OVERLAPS=0.6,
                                                MIN_SIZE_SIM=0.6,
                                                TEXT_PROPOSALS_MIN_SCORE=0.1,
                                                TEXT_PROPOSALS_NMS_THRESH=0.3,
                                                # 文本行之间测iou值
                                                TEXT_LINE_NMS_THRESH=0.99,
                                                LINE_MIN_SCORE=0.1,
                                                # 对检测的文本行进行向左延伸
                                                leftAdjustAlph=0.01,
                                                # 对检测的文本行进行向右延伸
                                                rightAdjustAlph=0.01)

                    if billModel == '' or billModel == '通用OCR':
                        result = union_rbox(result, 0.2)
                        res = [{'text': x['text'],
                                'name': str(i),
                                'box': {'cx': x['cx'],
                                        'cy': x['cy'],
                                        'w': x['w'],
                                        'h': x['h'],
                                        'angle': x['degree']
                                        }
                                } for i, x in enumerate(result)]
                        # 修正box
                        res = adjust_box_to_origin(img, angle, res)

                    elif billModel == '火车票':
                        res = trainTicket.trainTicket(result)
                        res = res.res
                        res = [{'text': res[key], 'name': key, 'box': {}} for key in res]

                    elif billModel == '身份证':

                        res = idcard.idcard(result)
                        res = res.res
                        res = [{'text': res[key], 'name': key, 'box': {}} for key in res]

                    os.remove(filelock)
                    break
        timeTake = time.time() - t

        return json.dumps({'res': res, 'timeTake': round(timeTake, 4)}, ensure_ascii=False)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5100)
