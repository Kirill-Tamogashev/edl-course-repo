import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18, resnet101, ResNet101_Weights


class DistillResNet101(nn.Module):
    def __init__(self, get_featuresc):
        super().__init__()

        self.resnet = resnet101(pretrained=False)

        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3)
        self.resnet.maxpool = nn.Identity()

        self.resnet.layer3 = nn.Conv2d(512, 1024, kernel_size=3)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)




