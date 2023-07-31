import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchsummary import summary
from matplotlib import pyplot as plt
from copy import deepcopy
import numpy as np
import cv2
import arch
import dataset


def getLearningRate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    

def metricBatch(output, target):
    pred = output.argmax(1, keepdim=True)
    corrects = pred.eq(target.view_as(pred)).sum().item()
    return corrects


def lossBatch(loss_func, output, target, opt=None):
    loss = loss_func(output, target)
    metric_b = metricBatch(output, target)

    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss.item(), metric_b


def lossEpoch(model, loss_func, dataset_dl, sanity_check=False, opt=None):
    running_loss = 0.0
    running_metric = 0.0
    len_data = len(dataset_dl.dataset)

    for xb, yb in dataset_dl:
        xb = xb.to('cuda')
        yb = yb.to('cuda')
        output = model(xb)

        loss_b, metric_b = lossBatch(loss_func, output, yb, opt)

        running_loss += loss_b
        
        if metric_b is not None:
            running_metric += metric_b
        
        if sanity_check is True:
            break

    loss = running_loss / len_data
    metric = running_metric / len_data

    return loss, metric


def train():
    if torch.cuda.is_available():
        device = torch.device("cuda")

    pathToData = "/home/woo/Desktop/job/Vision_Study/Paper_Implementation/data"
    batchSize = 32
    model = arch.ResNet152().to(device)     # 모델 정의

    trainTransform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
    ])

    valTransform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    trainData = datasets.STL10(pathToData, split='train', download=False, transform=trainTransform)
    valData = datasets.STL10(pathToData, split='test', download=False, transform=valTransform)

    trainLoader = DataLoader(trainData, batch_size=batchSize, shuffle=True)
    valLoader = DataLoader(valData, batch_size=batchSize, shuffle=False)

    epochsNum = 100
    lossFunction = nn.CrossEntropyLoss()        # 손실함수 정의
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)        # 옵티마이저 정의
    learningRateScheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)        # learning rate scheduler 정의
    sanityCheck = False
    pathToWeights = './models/weights.pt'

    try:
        if not os.path.exists('./models'):
            os.makedirs('./models')
    except OSerror:
        print('Error')

    lossHistory = {'train': [], 'val': []}
    metricHistory = {'train': [], 'val': []}

    bestModelWeights = deepcopy(model.state_dict())
    bestLoss = float('inf')

    summary(model, input_size=(3, 224, 224), batch_size=batchSize, device='cuda')

    startTime = time.time()

    for epoch in range(epochsNum):
        epochStartTime = time.time()

        curLR = getLearningRate(optimizer)
        print(f'Epoch {epoch + 1}/{epochsNum}, current lr = {curLR}')

        model.train()
        trainLoss, trainMetric = lossEpoch(model, lossFunction, trainLoader, sanity_check=False, opt=optimizer)
        lossHistory['train'].append(trainLoss)
        metricHistory['train'].append(trainMetric)

        model.eval()
        with torch.no_grad():
            valLoss, valMetric = lossEpoch(model, lossFunction, valLoader, sanityCheck)
        lossHistory['val'].append(valLoss)
        metricHistory['val'].append(valMetric)

        if valLoss < bestLoss:
            best_loss = valLoss
            bestModelWeights = deepcopy(model.state_dict())
            torch.save(model.state_dict(), pathToWeights)

        learningRateScheduler.step(valLoss)

        epochEndTime = time.time()
        print(f'train loss: {trainLoss:.6f}, val loss: {valLoss:.6f}, accuracy: {100 * valMetric:.2f}%, epoch time: {(epochEndTime - epochStartTime) / 60:.2f} min')

        if epoch // 10 == 0:
            torch.cuda.empty_cache()

    endTime = time.time()

    print(f'Training complete in {(endTime - startTime) / 60:.2f} min')
    

if __name__ == "__main__":
    train()