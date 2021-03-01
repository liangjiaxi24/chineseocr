from config import *
from main import TextOcrModel


filelock = 'file.lock'
if os.path.exists(filelock):
    os.remove(filelock)

if yoloTextFlag == 'keras' or AngleModelFlag == 'tf' or ocrFlag == 'keras':
    if GPU:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUID)
        import tensorflow as tf
        from keras import backend as K

        config = tf.ConfigProto()
        config.gpu_options.allocator_type = 'BFC'
        config.gpu_options.per_process_gpu_memory_fraction = 0.3  ## GPU最大占用量
        config.gpu_options.allow_growth = True  ##GPU是否可动态增加
        K.set_session(tf.Session(config=config))
        K.get_session().run(tf.global_variables_initializer())

    else:
        ##CPU启动
        os.environ["CUDA_VISIBLE_DEVICES"] = ''

if yoloTextFlag == 'opencv':
    scale, maxScale = IMGSIZE
    from text.opencv_dnn_detect import text_detect
elif yoloTextFlag == 'darknet':
    scale, maxScale = IMGSIZE
    from text.darknet_detect import text_detect
elif yoloTextFlag == 'keras':
    scale, maxScale = IMGSIZE[0], 2048
    from text.keras_detect import text_detect
else:
    print("err,text engine in keras\opencv\darknet")

from text.opencv_dnn_detect import angle_detect

if ocr_redis:
    ##多任务并发识别
    from apphelper.redisbase import redisDataBase

    ocr = redisDataBase().put_values
else:
    from crnn.keys import alphabetChinese, alphabetEnglish

    if ocrFlag == 'keras':
        from crnn.network_keras import CRNN

        if chineseModel:
            alphabet = alphabetChinese
            if LSTMFLAG:
                ocrModel = ocrModelKerasLstm
            else:
                ocrModel = ocrModelKerasDense
        else:
            ocrModel = ocrModelKerasEng
            alphabet = alphabetEnglish
            LSTMFLAG = True

    elif ocrFlag == 'torch':
        from crnn.network_torch import CRNN

        if chineseModel:
            alphabet = alphabetChinese
            if LSTMFLAG:
                ocrModel = ocrModelTorchLstm
            else:
                ocrModel = ocrModelTorchDense

        else:
            ocrModel = ocrModelTorchEng
            alphabet = alphabetEnglish
            LSTMFLAG = True
    elif ocrFlag == 'opencv':
        from crnn.network_dnn import CRNN

        ocrModel = ocrModelOpencv
        alphabet = alphabetChinese
    else:
        print("err,ocr engine in keras\opencv\darknet")

    nclass = len(alphabet) + 1
    if ocrFlag == 'opencv':
        crnn = CRNN(alphabet=alphabet)
    else:
        crnn = CRNN(32, 1, nclass, 256, leakyRelu=False, lstmFlag=LSTMFLAG, GPU=GPU, alphabet=alphabet)
    if os.path.exists(ocrModel):
        crnn.load_weights(ocrModel)
    else:
        print("download model or tranform model with tools!")

    ocr = crnn.predict_job

model = TextOcrModel(ocr, text_detect, angle_detect)
billList = ['通用OCR', '火车票', '身份证']
