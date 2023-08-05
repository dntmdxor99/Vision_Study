import torch
import torch.nn as nn
import torch.nn.functional as F


class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()



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
        