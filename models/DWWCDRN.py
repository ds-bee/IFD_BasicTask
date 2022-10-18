import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super().__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),  # 这个卷积操作是不会改变w h的
            nn.BatchNorm2d(outchannel))
        self.right = shortcut
        self.sigmoid = nn.Sigmoid()
        # 加入自定义系数
        # medium = torch.randn((1), requires_grad=True)
        # self.register_parameter("medium", torch.nn.Parameter(medium))
        # print(self._parameters)

    def forward(self, input):
        out = self.left(input)
        residual = input if self.right is None else self.right(input)
        # dcw = (self.medium)*residual if self.medium != 0 else 1
        out += residual                 # add Residual
        # 动态权重池化得到池化向量；
        # 这里一定是改变尺寸后的即residual的size(2)或者size(3)；
        v = residual.size(2)    # 获得特征图的宽度
        dwpool = nn.MaxPool2d(kernel_size=[1, v], stride=1)
        dcw = dwpool(residual)
        dcw = self.sigmoid(dcw)
        # print('dasda is:',dcw.size())
        dc = residual*dcw
        out += dc
        return F.relu(out)


class DwwcDrn(nn.Module):
    def __init__(self, num_class=5, num=3):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1))
        self.layer1 = self._make_layer(64, 128, 1)
        self.layer2 = self._make_layer(128, 256, 1, stride=2)
        self.layer3 = self._make_layer(256, 512, 1, stride=2)
        self.layer4 = self._make_layer(512, 512, 1, stride=2)
        self.layer5 = self._make_layer(512, 1024, 1, stride=2)
        self.fc = nn.Sequential(
            nn.Linear(256, 80),
            nn.Linear(80, num_class))

    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        shortcut = nn.Sequential(
                   nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
                   nn.BatchNorm2d(outchannel))
        layers = [ResidualBlock(inchannel, outchannel, stride, shortcut)]
        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.pre(input)
        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        # x = self.layer5(x)
        x = F.avg_pool2d(x, 8)  # 如果图片大小为224 ，经过多个ResidualBlock到这里刚好为7，所以做一个池化，为1，
        # print(x.size(0),x.size(1),x.size(2),x.size(3))
        # x = F.adaptive_avg_pool2d(x, (x.size(0),x.size(1),1,1))
        # print(x.shape)          # 所以如果图片大小小于224，都可以传入的，因为经过7的池化，肯定为1，但是大于224则不一定
        x = x.view(x.size(0), -1)
        # print(x.size())
        return self.fc(x)


if __name__ == "__main__":

    avg = 14
    import torch
    from torch.autograd import Variable
    x = torch.randn(3, 64, 64)
    net = DwwcDrn()
    x = Variable(torch.unsqueeze(x, dim=0).float(), requires_grad=False)
    print(x.shape)
    y = net(x)
    i = 3
    j = 6
    trans = nn.Sequential(
                nn.Conv2d(i, j, 1, stride=1, bias=False),
                nn.BatchNorm2d(j))
    net = ResidualBlock(inchannel=3, outchannel=6, shortcut=trans)
    y = net(x)
    print(y.shape)
