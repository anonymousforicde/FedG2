# !/usr/bin/python3
# -*- coding:utf-8 -*-


import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, width_mult=1, num_classes=10, ratio=0.0):
        super(AlexNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.ReLU(inplace=True),
            nn.Dropout(ratio),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Dropout(ratio),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.Dropout(ratio),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.Dropout(ratio),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Dropout(ratio),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(256 * 3 * 3, 1024),
            nn.Dropout(ratio),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Dropout(ratio),
        ) 
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        # print(x.shape)
        x = x.view(-1, 256 * 3 * 3)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


