import os
import sys


sys.path.insert(0, os.path.dirname(os.path.dirname(os.getcwd())))

import numpy as np
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data
from warpctc_pytorch import CTCLoss
from train.ocr.dataset import PathDataset, randomSequentialSampler, alignCollate
from glob import glob
from sklearn.model_selection import train_test_split
from config import ocrModel, LSTMFLAG, GPU
from crnn.util import strLabelConverter
from train.ocr.dataset import resizeNormalize
from crnn.util import loadData
from train.ocr.generic_utils import Progbar
from crnn.network_torch import CRNN
from crnn.keys import alphabetChinese, alphabetEnglish


roots = glob('../data/ocr/*/*.jpg')

# 此处未考虑字符平衡划分
trainP, testP = train_test_split(roots, test_size=0.1)

traindataset = PathDataset(trainP, alphabetEnglish)
testdataset = PathDataset(testP, alphabetEnglish)


batchSize = 8
workers = 1
imgH = 32
imgW = 280
keep_ratio = True
cuda = True
ngpu = 1
nh = 256

sampler = randomSequentialSampler(traindataset, batchSize)
train_loader = torch.utils.data.DataLoader(
    traindataset, batch_size=batchSize,
    shuffle=False, sampler=None,
    num_workers=int(workers),
    collate_fn=alignCollate(imgH=imgH, imgW=imgW, keep_ratio=keep_ratio))


train_iter = iter(train_loader)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


model = CRNN(32, 1, len(alphabetEnglish) + 1, 256, 1, lstmFlag=LSTMFLAG)
model.apply(weights_init)
preWeightDict = torch.load(ocrModel, map_location=lambda storage, loc: storage)  ##加入项目训练的权重

modelWeightDict = model.state_dict()


for k, v in preWeightDict.items():
    name = k.replace('module.', '')  # remove `module.`
    if 'rnn.1.embedding' not in name:  ##不加载最后一层权重
        modelWeightDict[name] = v

model.load_state_dict(modelWeightDict)

lr = 0.1
optimizer = optim.Adadelta(model.parameters(), lr=lr)
converter = strLabelConverter(''.join(alphabetEnglish))
criterion = CTCLoss()

image = torch.FloatTensor(batchSize, 3, imgH, imgH)
text = torch.IntTensor(batchSize * 5)
length = torch.IntTensor(batchSize)

if torch.cuda.is_available():
    model.cuda()
    model = torch.nn.DataParallel(model, device_ids=[0])  ##转换为多GPU训练模型
    image = image.cuda()
    criterion = criterion.cuda()


def trainBatch(net, criterion, optimizer, cpu_images, cpu_texts):
    # data = train_iter.next()
    # cpu_images, cpu_texts = data
    batch_size = cpu_images.size(0)
    loadData(image, cpu_images)
    t, l = converter.encode(cpu_texts)

    loadData(text, t)
    loadData(length, l)
    preds = net(image)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    # print(preds.shape, preds_size.shape)
    # print(preds_size)
    cost = criterion(preds, text, preds_size, length) / batch_size
    net.zero_grad()
    cost.backward()
    optimizer.step()
    # print(cost)
    return cost


def predict(im):
    """
    预测
    """
    image = im.convert('L')
    scale = image.size[1] * 1.0 / 32
    w = image.size[0] / scale
    w = int(w)
    transformer = resizeNormalize((w, 32))

    image = transformer(image)
    if torch.cuda.is_available():
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)
    preds = model(image)
    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)
    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    return sim_pred


def val(net, dataset, max_iter=100):
    for p in net.parameters():
        p.requires_grad = False
    net.eval()
    i = 0
    n_correct = 0
    N = len(dataset)

    max_iter = min(max_iter, N)
    for i in range(max_iter):
        im, label = dataset[np.random.randint(0, N)]
        if im.size[0] > 1024:
            continue

        pred = predict(im)
        if pred.strip() == label:
            n_correct += 1

    accuracy = n_correct / float(max_iter)
    return accuracy


def train_model():
    nepochs = 10
    acc = 0

    interval = len(train_loader) // 2  ##评估模型

    for i in range(nepochs):
        print('epoch:{}/{}'.format(i, nepochs))
        n = len(train_loader)
        pbar = Progbar(target=n)
        train_iter = iter(train_loader)
        loss = 0

        for j in range(n):
            for p in model.named_parameters():
                p[1].requires_grad = True
                if 'rnn.1.embedding' in p[0]:
                    p[1].requires_grad = True
                else:
                    p[1].requires_grad = False  ##冻结模型层

            model.train()
            cpu_images, cpu_texts = next(train_iter)

            cost = trainBatch(model, criterion, optimizer, cpu_images, cpu_texts)

            loss += cost.data.numpy()

            if (j + 1) % interval == 0:
                curAcc = val(model, testdataset, max_iter=1024)
                if curAcc > acc:
                    acc = curAcc
                    torch.save(model.state_dict(), 'new_modellstm.pth')

            pbar.update(j + 1, values=[('loss', loss / ((j + 1) * batchSize)), ('acc', acc)])


# print(model.eval())


N = len(testdataset)
print(N)

for i in range(500):
    im, label = testdataset[np.random.randint(0, N)]
    pred = predict(im)
    print('true:{},pred:{}'.format(label, pred))
