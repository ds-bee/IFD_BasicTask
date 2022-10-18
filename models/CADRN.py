import torch.nn as nn
import torch.nn.functional as F

# class SEBlock(nn.Module):
#     """
#     Channel Attention
#     """


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return x * self.sigmoid(out)


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super().__init__()
        self.CAModule = ChannelAttention(outchannel)    ###
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),  # 这个卷积操作是不会改变w h的
            nn.BatchNorm2d(outchannel))
        self.right = shortcut

    def forward(self, input):
        out = self.left(input)
        out = self.CAModule(out)    ###
        residual = input if self.right is None else self.right(input)
        out += residual                 # add Residual
        return F.relu(out)


class CADRN(nn.Module):
    def __init__(self, num_class=5, num=3):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1))
        # 重复layer，每个layer都包含多个残差块 其中第一个残差会修改w和c，其他的残差块等量变换
        # 经过第一个残差块后大小为 w-1/s +1 （每个残差块包括left和right，而left的k = 3 p = 1，right的shortcut k=1，p=0）
        self.layer1 = self._make_layer(64, 128, num)  # s默认是1 ,所以经过layer1后只有channel变了
        self.layer2 = self._make_layer(128, 256, num, stride=2)  # w-1/s +1
        self.layer3 = self._make_layer(256, 512, num, stride=2)
        self.layer4 = self._make_layer(512, 512, num, stride=2)
        self.layer5 = self._make_layer(512, 1024, 1, stride=2)
        self.fc = nn.Sequential(
            # nn.Linear(1024, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 80),
            nn.Linear(80, num_class))

    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        # 刚开始两个channel可能不同，所以right通过shortcut把通道也变为out channel
        shortcut = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
            nn.BatchNorm2d(outchannel))                                   # 之所以这里的k = 1是因为，我们在ResidualBlock中的k =3,p=1所以最后得到的大小为(w+2-3/s +1)
                                                                          #  即(w-1 /s +1)，而这里的w = (w +2p-f)/s +1 所以2p -f = -1 如果p = 0则f = 1
        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))
        for i in range(1, block_num):                   # 之后的cahnnle同并且 w h也同，而经过ResidualBloc其w h不变，
            layers.append(ResidualBlock(outchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.pre(input)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # x = self.layer5(x)
        x = F.avg_pool2d(x, 7)  # 如果图片大小为224 ，经过多个ResidualBlock到这里刚好为7，所以做一个池化，为1，
        # print(x.shape)          # 所以如果图片大小小于224，都可以传入的，因为经过7的池化，肯定为1，但是大于224则不一定
        x = x.view(x.size(0), -1)
        return self.fc(x)
        # return x


if __name__== "__main__":
    import torch
    from torch.autograd import Variable
    x = torch.randn(3, 256, 256)
    # print(x.shape)
    net = CAResNet()
    x = Variable(torch.unsqueeze(x, dim=0).float(), requires_grad=False)
    # print(x.shape)  # [1,3, 224, 224]
    y = net(x)
    # print(y.shape)
