import torch
import torch.nn as nn
import torch.nn.functional as F


class GoogLeNet(nn.Module):
    def __init__(self, numClasses):
        super(GoogLeNet, self).__init__()

        self.training = True
        
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding = 3),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1),
            nn.LocalResponseNorm(2),

            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(64, 192, kernel_size = 3, padding = 1),
            nn.ReLU(True),
            nn.LocalResponseNorm(2),
            nn.MaxPool2d(kernel_size=3, stride= 2, padding = 1)
        )        


        self.inception_3 = nn.Sequential(
            Inception(192, 64, 96, 128, 16, 32, 32),
            Inception(256, 128, 128, 192, 32, 96, 64)
        )

        
        self.inception_4a = Inception(480, 192, 96, 208, 16, 48, 64)

        if self.training:
            self.aux_1 = AuxClassifier(512, numClasses)

        self.inception_4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception_4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception_4d = Inception(512, 112, 144, 288, 32, 64, 64)

        if self.training:
            self.aux_2 = AuxClassifier(528, numClasses)

        self.inception_4e = Inception(528, 256, 160, 320, 32, 128, 128)


        self.inception_5 = nn.Sequential(
            Inception(832, 256, 160, 320, 32, 128, 128),
            Inception(832, 384, 192, 384, 48, 128, 128)
        )

        self.maxPool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        self.avgPool = nn.AvgPool2d(kernel_size = 7, stride = 1)
        self.dropOut = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, numClasses)

    
    def forward(self, x):
        x = self.conv(x)    
        x = self.inception_3(x)
        x = self.maxPool(x)
        x = self.inception_4a(x)

        if self.training:
            out_1 = self.aux_1(x)

        x = self.inception_4b(x)
        x = self.inception_4c(x)
        x = self.inception_4d(x)

        if self.training:
            out_2 = self.aux_2(x)

        x = self.inception_4e(x)
        x = self.maxPool(x)
        x = self.inception_5(x)
        
        x = self.avgPool(x)
        x = x.view(x.shape[0], -1)
        x = self.dropOut(x)
        x = self.fc(x)

        if self.training:
            return [x, out_1, out_2]
        else:
            return x
        
    
    def setTraining(self):
        self.training = True

    def setEval(self):
        self.training = False


class AuxClassifier(nn.Module):
    def __init__(self, inChannel, numClasses):
        super(AuxClassifier, self).__init__()

        self.conv = nn.Sequential(
            nn.AvgPool2d(kernel_size = 5, stride = 3),
            nn.Conv2d(inChannel, 128, kernel_size=1),
            nn.ReLU(True)
        )

        self.fc = nn.Sequential(
            nn.Linear(4 * 4 * 128, 1024),
            nn.ReLU(True),
            nn.Dropout(0.7),
            nn.Linear(1024, numClasses)
        )

    def forward(self, x):
        out = self.conv(x)
        out = out.view([out.shape[0], -1])
        out = self.fc(out)

        return out


class Inception(nn.Module):
    def __init__(self, inChannel, out_11, outReduce_33, out_33, outReduce_55, out_55, outMaxPool_11):
        super(Inception, self).__init__()

        self.branch_1 = nn.Sequential(
            nn.Conv2d(inChannel, out_11, kernel_size = 1),
            nn.ReLU(True)
        )

        self.branch_2 = nn.Sequential(
            nn.Conv2d(inChannel, outReduce_33, kernel_size = 1),
            nn.ReLU(True),
            nn.Conv2d(outReduce_33, out_33, kernel_size = 3, padding = 1),
            nn.ReLU(True)
        )

        self.branch_3 = nn.Sequential(
            nn.Conv2d(inChannel, outReduce_55, kernel_size = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(outReduce_55, out_55, kernel_size = 5, padding = 2),
            nn.ReLU(True)
        )

        self.branch_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size = 3, padding = 1, stride = 1),     # stride default = kernel size와 같음
            nn.Conv2d(inChannel, outMaxPool_11, kernel_size = 1),
            nn.ReLU(True)
        )

    def forward(self, x):
        out_1 = self.branch_1(x)
        out_2 = self.branch_2(x)
        out_3 = self.branch_3(x)
        out_4 = self.branch_4(x)

        return torch.cat([out_1, out_2, out_3, out_4], dim = 1)
        