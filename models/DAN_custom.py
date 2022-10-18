import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import models.mmd_custom as mmd_custom
import torch
from models.lenet_custom import LeNet

__all__ = ['ResNet', 'resnet50', 'DANNet', 'DANNet_test']

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


def conv3x3(in_planes, out_planes, stride=1,groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # x=self.fc(x)

        return x


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model_name.
    Args:
        pretrained (bool): If True, returns a model_name pre-trained on ImageNet
    """
    # model_name = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    model = ResNet(Bottleneck, [1, 1, 1, 1], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


class DANNet(nn.Module):

    def __init__(self, num_classes=4, num_resblock=1):
        super(DANNet, self).__init__()
        self.sharedNet = resnet50(False)    # raopt:True->False   # 2048
        # self.sharedNet = CAResNet(num=num_resblock, img_channel=3)
        self.cls_fc2 = nn.Sequential(
                       nn.Linear(512, num_classes))
        self.class_coefficient = torch.nn.Parameter(torch.ones([5, 1]), requires_grad=True)

    def forward(self, source, target):
        loss = 0.
        source = self.sharedNet(source)
        sour = source.clone()   # temporary save
        if self.training:       # belong to module.train(), default=True;except eval()
            target = self.sharedNet(target)
            targ = target.clone()
        source = self.cls_fc2(source)
        target = self.cls_fc2(target)
        if self.training:
            loss = mmd_custom.mmd_rbf_noaccelerate(sour, targ, source, target, self.class_coefficient)
            # loss = mmd_custom.mmd_rbf_test(sour, targ, source, target)
        return source, loss


class DANNet_test(nn.Module):

    def __init__(self, num_classes=4):
        super(DANNet_test, self).__init__()
        self.sharedNet = LeNet()
        self.cls_fc2 = nn.Sequential(
                       nn.Linear(1024, num_classes))
        self.class_coefficient = torch.nn.Parameter(torch.ones([5, 1]), requires_grad=True)

    def forward(self, x):
        source = x[0]
        target = x[-1]
        loss = 0.
        source = self.sharedNet(source)
        sour = source.clone()   # temporary save
        target = self.sharedNet(target)
        if self.training:       # belong to module.train(), default=True;except eval()
            targ = target.clone()
        source = self.cls_fc2(source)
        target = self.cls_fc2(target)
        if self.training:
            mmd_loss = mmd_custom.mmd_rbf_noaccelerate(sour, targ, source, target, self.class_coefficient)   # raopt
        else:
            mmd_loss = 0.
        return source, mmd_loss
