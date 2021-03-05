import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.getcwd())))
from glob import glob
from sklearn.model_selection import train_test_split
from train.ocr.generic_utils import Progbar
from crnn.keys import alphabetChinese, alphabetEnglish
from train.ocr.dataset import PathDataset, randomSequentialSampler, alignCollate
from crnn.util import loadData
from warpctc_pytorch import CTCLoss
import torch.optim as optim


import torch
import numpy as np
import torch.nn as nn
from collections import OrderedDict
from torch.autograd import Variable
from crnn.utils import strLabelConverter as ValstrLabelConverter
from crnn.util import strLabelConverter
from crnn.utils import resizeNormalize


class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)
        return output


class CRNN(nn.Module):
    def __init__(self, imgH, nc, nclass, nh, leakyRelu=False, lstmFlag=True, GPU=False, alphabet=None):
        """
        是否加入lstm特征层
        """
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]
        self.lstmFlag = lstmFlag
        self.GPU = GPU
        self.alphabet = alphabet
        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16
        
        self.cnn = cnn
        if self.lstmFlag:
            self.rnn = nn.Sequential(
                BidirectionalLSTM(512, nh, nh),
                BidirectionalLSTM(nh, nh, nclass))
        else:
            self.linear = nn.Linear(nh*2, nclass)

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        if self.lstmFlag:
           # rnn features
           output = self.rnn(conv)
           T, b, h = output.size()
           output = output.view(T, b, -1)
        else:
             T, b, h = conv.size()
             t_rec = conv.contiguous().view(T * b, h)
             output = self.linear(t_rec)  # [T * b, nOut]
             output = output.view(T, b, -1)
        return output
    
    def load_weights(self, path):
        trainWeights = torch.load(path, map_location=lambda storage, loc: storage)
        modelWeights = OrderedDict()
        for k, v in trainWeights.items():
            name = k.replace('module.', '') # remove `module.`
            modelWeights[name] = v      
        self.load_state_dict(modelWeights)
        if torch.cuda.is_available() and self.GPU:
            self.cuda()
        self.eval()

    def predict(self, image):
        resize_normalize = resizeNormalize((32, 32))
        image = resize_normalize(image)

        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        if torch.cuda.is_available() and self.GPU:
           image = image.cuda()
        else:
           image = image.cpu()
            
        image = image.view(1, 1, *image.size())
        image = Variable(image)
        if image.size()[-1] < 8:
            return ''
        preds = self(image)
        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)

        converter = ValstrLabelConverter(self.alphabet)
        raw = converter.decode(preds)

        return raw
    
    def predict_job(self,boxes):
        n = len(boxes)
        for i in range(n):
            
            boxes[i]['text'] = self.predict(boxes[i]['img'])
            
        return boxes
        
    def predict_batch(self, boxes, batch_size=1):
        """
        predict on batch
        """

        N = len(boxes)
        res = []
        imgW = 0
        batch = N//batch_size
        if batch*batch_size != N:
            batch += 1
        for i in range(batch):
            tmpBoxes = boxes[i*batch_size:(i+1)*batch_size]
            imageBatch = []
            imgW = 0
            for box in tmpBoxes:
                img = box['img']
                image = resizeNormalize(img,32)
                h, w = image.shape[:2]
                imgW = max(imgW, w)
                imageBatch.append(np.array([image]))
                
            imageArray = np.zeros((len(imageBatch), 1, 32, imgW), dtype=np.float32)
            n = len(imageArray)
            for j in range(n):
                _, h, w = imageBatch[j].shape
                imageArray[j][:, :, :w] = imageBatch[j]
            
            image = torch.from_numpy(imageArray)
            image = Variable(image)
            if torch.cuda.is_available() and self.GPU:
                image = image.cuda()
            else:
                image = image.cpu()
                
            preds = self(image)
            preds = preds.argmax(2)
            n = preds.shape[1]
            for j in range(n):
                res.append(strLabelConverter(preds[:, j], self.alphabet))

        for i in range(N):
            boxes[i]['text'] = res[i]
        return boxes

    def train_model(self):
        batchSize = 8
        workers = 1
        imgH = 32
        imgW = 280
        keep_ratio = True
        nepochs = 10
        acc = 0
        lr = 0.1

        image = torch.FloatTensor(batchSize, 3, imgH, imgH)
        text = torch.IntTensor(batchSize * 5)
        length = torch.IntTensor(batchSize)
        converter = strLabelConverter(''.join(alphabetEnglish))
        optimizer = optim.Adadelta(self.parameters(), lr=lr)

        roots = glob('../data/ocr/*/*.jpg')
        # 此处未考虑字符平衡划分
        trainP, testP = train_test_split(roots, test_size=0.1)
        traindataset = PathDataset(trainP, alphabetEnglish)
        testdataset = PathDataset(testP, alphabetEnglish)
        criterion = CTCLoss()

        train_loader = torch.utils.data.DataLoader(
            traindataset, batch_size=batchSize,
            shuffle=False, sampler=None,
            num_workers=int(workers),
            collate_fn=alignCollate(imgH=imgH, imgW=imgW, keep_ratio=keep_ratio))
        interval = len(train_loader) // 2  ##评估模型

        def trainBatch(net, criterion, optimizer, cpu_images, cpu_texts):
            batch_size = cpu_images.size(0)
            loadData(image, cpu_images)
            t, l = converter.encode(cpu_texts)

            loadData(text, t)
            loadData(length, l)
            preds = net(image)
            preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
            cost = criterion(preds, text, preds_size, length) / batch_size
            net.zero_grad()
            cost.backward()
            optimizer.step()
            return cost

        def val(net, dataset, max_iter=100):
            for p in net.parameters():
                p.requires_grad = False
            net.eval()
            n_correct = 0
            N = len(dataset)
            max_iter = min(max_iter, N)
            for i in range(max_iter):
                im, label = dataset[np.random.randint(0, N)]
                if im.size[0] > 1024:
                    continue
                pred = self.predict(im)
                if pred.strip() == label:
                    n_correct += 1
            accuracy = n_correct / float(max_iter)
            return accuracy

        if torch.cuda.is_available():
            self.cuda()
            # model = torch.nn.DataParallel(model, device_ids=[0])  ##转换为多GPU训练模型
            image = image.cuda()
            criterion = criterion.cuda()

        for i in range(nepochs):
            # mark = 1
            print('epoch:{}/{}'.format(i, nepochs))
            n = len(train_loader)
            pbar = Progbar(target=n)
            train_iter = iter(train_loader)
            loss = 0

            for j in range(n):
                for name, params in self.named_parameters():
                    params.requires_grad = True

                    if 'rnn.1.embedding' in name:
                        params.requires_grad = True
                        # data = params.data.cuda().data.cpu().numpy()
                        # if mark == 1:
                        #     f = open("./mark_parms/{}_epoch_{}.txt".format(name, i), "w")
                        #     f.write(name)
                        #     f.write("\n")
                        #     f.write(str(data))
                        #     f.close()
                        # mark = 2

                    else:
                        params.requires_grad = False  ##冻结模型层

                self.train()
                cpu_images, cpu_texts = next(train_iter)

                cost = trainBatch(self, criterion, optimizer, cpu_images, cpu_texts)

                loss += cost.data.numpy()

                if (j + 1) % interval == 0:
                    curAcc = val(self, testdataset, max_iter=1024)
                    if curAcc > acc:
                        acc = curAcc
                    torch.save(self.state_dict(), 'new_modellstm.pth')

                pbar.update(j + 1, values=[('loss', loss / ((j + 1) * batchSize)), ('acc', acc)])


if __name__ == '__main__':
    from crnn.keys import alphabetChinese
    from PIL import Image

    ocr_path = './models/ocr-lstm.pth'
    # ocr_path = './models/new_modellstm.pth'
    alphabet = alphabetChinese
    nclass = len(alphabet) + 1

    crnn_model = CRNN(32, 1, nclass, 256, leakyRelu=False, lstmFlag=True, GPU=False, alphabet=alphabet)
    crnn_model.load_weights(ocr_path)

    # img = Image.open("2.png")
    # img = img.convert('L')
    # raw = crnn_model.predict(img)
    # print(raw)

    crnn_model.train_model()
