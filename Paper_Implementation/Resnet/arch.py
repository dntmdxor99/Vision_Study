import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride        # stride = 2라면 Downsampling을 적용하는 것임

        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size = 3, stride = self.stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(self.out_channels, BasicBlock.expansion * self.out_channels, kernel_size = 3, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(self.out_channels * BasicBlock.expansion)

        if self.stride != 1 or self.in_channels != BasicBlock.expansion * self.out_channels:
            # stride가 1이 아니라면, downsampling을 한 것임
            # in_channels가 BasicBlock.expansion * self.out_channels가 아니라면, channel을 맞춰주기 위해 1x1 conv를 적용해야 함
            self.shortcut = nn.Sequential(
                nn.Conv2d(self.in_channels, BasicBlock.expansion * self.out_channels, kernel_size = 1, stride = self.stride, bias = False),
                nn.BatchNorm2d(BasicBlock.expansion * self.out_channels)
            )
        else:
            self.shortcut = nn.Sequential()
            

    def forward(self, x):
        identity = x        ## 원본임
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.shortcut(x)

        out += identity
        out = self.relu(out)

        return out
    

class BottleNeck(nn.Module):
    
    expansion = 4

    def __init__(self, in_channels, out_channels, stride = 1):
        super(BottleNeck, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.relu = nn.ReLU(inplace = True)

        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size = 3, stride = self.stride, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(self.out_channels)

        self.conv3 = nn.Conv2d(self.out_channels, BottleNeck.expansion * self.out_channels, kernel_size = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(BottleNeck.expansion * self.out_channels)

        if self.stride != 1 or self.in_channels != BottleNeck.expansion * self.out_channels:
            # stride가 1이 아니라면, downsampling을 한 것임
            # in_channels가 BasicBlock.expansion * self.out_channels가 아니라면, channel을 맞춰주기 위해 1x1 conv를 적용해야 함
            self.shortcut = nn.Sequential(
                nn.Conv2d(self.in_channels, BottleNeck.expansion * self.out_channels, kernel_size = 1, stride = self.stride, bias = False),
                nn.BatchNorm2d(BottleNeck.expansion * self.out_channels)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(x)
        out = self.bn3(out)
        
        identity = self.shortcut(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()

        self.block = block
        self.layers = layers
        self.num_classes = num_classes

        self.in_channels = 64       # 논문에 적혀있음
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace = True)
        self.maxPool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        self.conv2 = self.makeLayer(block, 64, layers[0], 1)
        self.conv3 = self.makeLayer(block, 128, layers[1], 2)
        self.conv4 = self.makeLayer(block, 256, layers[2], 2)
        self.conv5 = self.makeLayer(block, 512, layers[3], 2)

        self.avgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, self.num_classes)

    
    def makeLayer(self, block, out_channels, num_blocks, stride = 1):
        strides = [stride] + [1] * (num_blocks - 1)      # 첫번째 block만 stride를 적용하고, 나머지는 1로 적용
        layers = []

        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)


    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxPool(out)

        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)

        out = self.avgPool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(BottleNeck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(BottleNeck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(BottleNeck, [3, 8, 36, 3])