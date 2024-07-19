import torch
import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self, dim, in_channels, num_actions) -> None:
        super(ConvNet, self).__init__()

        self.cnn_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.cnn_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.cnn_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.flatten = nn.Flatten()
        self.fc_layer1 = nn.Sequential(
            nn.Linear(64*8*8, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.fc_layer2 = nn.Sequential(
            nn.Linear(256, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.fc_layer3 = nn.Sequential(
            nn.Linear(32, num_actions),
            nn.Sigmoid()
        )
        
        self.conv1 = nn.Conv2d(in_channels, 32, 8, 4)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 4, 3)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.conv3_bn = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(64*8*8, 256)
        self.fc1_bn = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 32)
        self.fc2_bn = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, num_actions)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.cnn_layer1(x)
        x = self.cnn_layer2(x)
        x = self.cnn_layer3(x)
        x = self.flatten(x)
        x = self.fc_layer1(x)
        x = self.fc_layer2(x)
        x = self.fc_layer3(x)
        return x
    