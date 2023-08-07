import os
import arch
import time

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets, transforms

from torchsummary import summary


def train():
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    batchSize = 128
    epochs = 1
    learningRate = 1e-3


    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(224, antialias=True, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    trainData = datasets.STL10('./data', split = 'train', download = False, transform=transform)
    trainLoader = DataLoader(trainData, batch_size = batchSize, shuffle = True, num_workers=2)

    valData = datasets.STL10('./data', split = 'test', download = False, transform=transform)
    valLoader = DataLoader(valData, batch_size = batchSize, shuffle = False, num_workers=2)

    model = arch.GoogLeNet(10).to(device)
    summary(model, (3, 224, 224), batch_size=batchSize, device=device)
    
    lossFunction = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = learningRate)

    bestLoss = float('inf')
    bestAcc = 0
    bestEpoch = 0

    LearningStartTime = time.time()
    for epoch in range(epochs):
        epochStartTime = time.time()

        print(f"Epoch : {epoch + 1}/{epochs}")

        model.train()
        model.setTraining()
        trainLen = len(trainLoader.dataset)
        trainLoss = 0
        trainAcc = 0
        for images, targets in trainLoader:
            images, targets = images.to(device), targets.to(device)

            out = model(images)     # List

            pred = out[0].argmax(1)
            acc = pred.eq(targets).sum().item()
    
            loss = [lossFunction(out[0], targets), lossFunction(out[1], targets), lossFunction(out[2], targets)]
            loss = loss[0] + loss[1] * 0.3 + loss[2] * 0.3

            trainLoss += loss.item()
            trainAcc += acc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        model.eval()
        model.setEval()
        valLen = len(trainLoader.dataset)
        valLoss = 0
        valAcc = 0
        with torch.no_grad():
            for images, targets in valLoader:
                images, targets = images.to(device), targets.to(device)

                out = model(images)     # List

                pred = out.argmax(1)
                acc = pred.eq(targets).sum().item()
        
                loss = lossFunction(out, targets)

                valLoss += loss.item()
                valAcc += acc
        
        trainLoss, trainAcc, valLoss, valAcc = trainLoss / trainLen, trainAcc / trainLen, valLoss / valLen, valAcc / valLen
        
        if valLoss < bestLoss:
            bestLoss = valLoss
            bestAcc = valAcc
            bestEpoch = epoch

        epochEndTime = time.time()
        print(f'train loss : {trainLoss:.6f}, train accuracy : {100 * trainAcc:.2f}, \n\
              val loss : {valLoss:.6f}, val accuracy : {100 * valAcc:.2f}, epoch time: {(epochEndTime - epochStartTime) / 60:.2f} min')
        
        if epoch // 10 == 0:
            torch.cuda.empty_cache()
    
    LearningEndTime = time.tiime()
    print(f'Training Complete in {(LearningEndTime - LearningStartTime) / 60:.2f} min')
    print(f'Best Val Loss : {bestLoss}, Best Val Accuracy : {bestAcc}, Epoch : {bestEpoch}')


if __name__ == "__main__":
    train()