import time
import torch
from torch import nn, optim
import sys

sys.path.append("..")
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')


class WtCnn(nn.Module):
    # def __init__(self, num_class=5):
    def __init__(self, num_class=10):
        super(WtCnn, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, padding=0, stride=2),

            nn.Conv2d(16, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, padding=0, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, padding=0, stride=2),

            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, padding=0, stride=2)
        )
        self.fcin_size = 64 * 4 * 4
        self.fc = nn.Sequential(
            nn.Linear(self.fcin_size, 512),
            nn.Sigmoid(),
            nn.Linear(512, 84),
            nn.Sigmoid(),
            nn.Linear(84, num_class)
        )


    def forward(self, img):
        # print(' ', img.shape[0])
        feature = self.conv1(img)
        feature = self.conv2(feature)
        feature = feature.view(feature.size(0), -1)
        # print(feature[0] - feature[1]) # 观察两张图的计算结果区别
        output = self.fc(feature)
        return output


if __name__ == '__main__':
    inp = torch.randn(2, 3, 64, 64)
    net = WtCnn()
    oup = net(inp)
    print(oup.shape)
