import os
import sys
import time
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms
from datasets_process.datasets_custom.dataset import MyDataSet
from utils.plot_diagram import *
import numpy as np
import seaborn as sns   
from sklearn.manifold import TSNE       # tsne
from matplotlib import pyplot as plt          


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


class caresnet(nn.Module):
    def __init__(self, num_class=5, num=3):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1))
        # 重复layer，每个layer都包含多个残差块 其中第一个残差会修改w和c，其他的残差块等量变换
        # 经过第一个残差块后大小为 w-1/s +1 （每个残差块包括left和right，而left的k = 3 p = 1，right的shortcut k=1，p=0）
        self.layer1 = self._make_layer(64, 128, num)  # s默认是1 ,所以经过layer1后只有channle变了
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
            nn.BatchNorm2d(outchannel))                                   # 之所以这里的k = 1是因为，我们在ResidualBlock中的k=3,p=1所以最后得到的大小为(w+2-3/s +1)
                                                                          #  即(w-1 /s +1)，而这里的w = (w +2p-f)/s +1 所以2p -f = -1 如果p = 0则f = 1
        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))
        for i in range(1, block_num):                   # 之后的cahnnle同并且 w h也同，而经过ResidualBloc其w h不变，
            layers.append(ResidualBlock(outchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, input):

        xp = self.pre(input)       
        x1 = self.layer1(xp)      
        x2 = self.layer2(x1)  
        x3 = self.layer3(x2)      
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        x = F.avg_pool2d(x4, 7)  # 如果图片大小为224 ，经过多个ResidualBlock到这里刚好为7，所以做一个池化，为1，
        # print(x.shape)          # 所以如果图片大小小于224，都可以传入的，因为经过7的池化，肯定为1，但是大于224则不一定
        x = x.view(x.size(0), -1)
        out_p =  self.fc(x)
        # return xp, x1, x2, x3, x4
        return xp, x4

n = 256                # 输入尺寸
ti = 1                  # 测试次数
batchsize = 500
num_class = 5
transform = transforms.Compose(
         [transforms.Resize([n, n]),
          transforms.ToTensor(),
          transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
load_PATH = './Model/DWB_Models/Cmor33_CWT_CADRN.pth'
classes = ('chi', 'cra', 'mis', 'nor', 'sur')
ts = MyDataSet('gear_labels', 'mixed_train', transform=transform)
test_loader = torch.utils.data.DataLoader(dataset=ts, batch_size=batchsize,
                                              shuffle=False, num_workers=0)  # 打开随机


def to_cpu_ndarray(obj):

    obj = obj.cpu()
    obj = obj.numpy()
    return obj


def tsne_plot_forlist(num_class, img, label, fn):

    res = img
    img = torch.cat(img,dim=0).reshape(240, 5)
    img = to_cpu_ndarray(img)
    tsne = TSNE(n_components=2)
    X_embedded = tsne.fit_transform(img)
    # 绘图设置：
    sns.set(style='white', rc={'figure.figsize':(7,7)})
    palette = sns.color_palette("bright", num_class)
    markers = ['o', 'v', '^', '<', '>']
    dirs = 'E:/AAA_LYB_RAOPT/Projects/My_Method/tsne_picture/A-240-mj-svg/'
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=label, style=label,
                                legend=True, palette=palette, markers=markers)
    # plt.savefig(dirs + fn+'.png')
    plt.show()


def tsne_plot(num_class, img, label, fn):

    print(img.shape)
    imgshape = img.shape
    img = img.reshape(imgshape[0], (imgshape[1]*imgshape[2])*imgshape[3])    
    tsne = TSNE(n_components=2)
    X_embedded = tsne.fit_transform(img)
    # 绘图设置：
    sns.set(style='white', rc={'figure.figsize':(7,7)})
    palette = sns.color_palette("bright", num_class)
    markers = ['o', 'v', '^', '<', '>']
    dirs = 'E:/AAA_LYB_RAOPT/Projects/My_Method/tsne_picture/A-240-mj-svg/'
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=label, style=label,
                                legend=True, palette=palette, markers=markers)
    # plt.savefig(dirs + fn+'.png')
    plt.show()


def main():

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = caresnet().to(device)
    net.load_state_dict(torch.load(load_PATH))
    # 整个测试集
    ti_num = []
    for i in range(ti):
        correct = 0
        total = 0
        with torch.no_grad():
            for i, data in enumerate(test_loader, 1):
                images, labels = data[0].to(device), data[1].to(device)
                labels = to_cpu_ndarray(labels.cpu())
                outputs = net(images)   # 多层输出则是tuple
                for i in range(len(outputs)):
                    x = to_cpu_ndarray(outputs[i])  # x取最后一个 
                fn = 'x4'
                tsne_plot(num_class, x, labels, fn)   


if __name__=='__main__':
    main()
