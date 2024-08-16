import torch
import torch.nn as nn

class MobileNetV1Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(MobileNetV1Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        return out

class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(MobileNetV1Block(32, 64, stride=1),
            MobileNetV1Block(64, 128, stride=2),
            MobileNetV1Block(128, 128, stride=1),
            MobileNetV1Block(128, 256, stride=2),
            MobileNetV1Block(256, 256, stride=1))

        self.conv3 = nn.Sequential(MobileNetV1Block(256, 512, stride=2),
            MobileNetV1Block(512, 512, stride=1),
            MobileNetV1Block(512, 512, stride=1),
            MobileNetV1Block(512, 512, stride=1),
            MobileNetV1Block(512, 512, stride=1))

        self.conv4 = nn.Sequential(MobileNetV1Block(512, 1024, stride=2),
            MobileNetV1Block(1024, 1024, stride=1),
            MobileNetV1Block(1024, 1024, stride=1))
           
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        return x
