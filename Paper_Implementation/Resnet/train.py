import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchsummary import summary
from matplotlib import pyplot as plt
import numpy as np
import cv2
import arch
import dataset


def train():
    pathToData = "/home/woo/Desktop/job/Vision_Study/Paper_Implementation/data"
    batchSize = 128


    if torch.cuda.is_available():
        device = torch.device("cuda")

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




if __name__ == "__main__":
    train()