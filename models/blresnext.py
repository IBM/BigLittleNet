# -*- coding: utf-8 -*-

# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import math

import torch
import torch.nn as nn

from ._model_urls import model_urls

__all__ = ['blresnext_model']


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, basewidth, cardinality, stride=1, downsample=None, last_relu=True):
        super(Bottleneck, self).__init__()

        D = int(math.floor(planes * (basewidth / 64.0))) // self.expansion
        C = cardinality

        self.conv1 = nn.Conv2d(inplanes, D * C, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(D * C)
        self.conv2 = nn.Conv2d(D * C, D * C, kernel_size=3, stride=stride,
                               padding=1, bias=False, groups=C)
        self.bn2 = nn.BatchNorm2d(D * C)
        self.conv3 = nn.Conv2d(D * C, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.last_relu = last_relu

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
        if self.last_relu:
            out = self.relu(out)

        return out


class bLModule(nn.Module):
    def __init__(self, block, in_channels, out_channels, blocks, basewidth, cardinality, alpha, beta, stride):
        super(bLModule, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.big = self._make_layer(block, in_channels, out_channels, blocks - 1,
                                    basewidth, cardinality, 2, last_relu=False)
        self.little = self._make_layer(block, in_channels, out_channels // alpha,
                                       max(1, blocks // beta - 1), basewidth * alpha, cardinality // alpha)
        self.little_e = nn.Sequential(
            nn.Conv2d(out_channels // alpha, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels))

        self.fusion = self._make_layer(block, out_channels, out_channels, 1, basewidth, cardinality, stride=stride)

    def _make_layer(self, block, inplanes, planes, blocks, basewidth, cardinality, stride=1, last_relu=True):
        downsample = []
        if stride != 1:
            downsample.append(nn.AvgPool2d(3, stride=2, padding=1))
        if inplanes != planes:
            downsample.append(nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False))
            downsample.append(nn.BatchNorm2d(planes))
        downsample = None if downsample == [] else nn.Sequential(*downsample)

        layers = []
        if blocks == 1:
            layers.append(block(inplanes, planes, basewidth, cardinality, stride=stride, downsample=downsample))
        else:
            layers.append(block(inplanes, planes, basewidth, cardinality, stride, downsample))
            for i in range(1, blocks):
                layers.append(block(planes, planes, basewidth, cardinality,
                                    last_relu=last_relu if i == blocks - 1 else True))

        return nn.Sequential(*layers)

    def forward(self, x):
        big = self.big(x)
        little = self.little(x)
        little = self.little_e(little)
        big = torch.nn.functional.interpolate(big, little.shape[2:])
        out = self.relu(big + little)
        out = self.fusion(out)

        return out


class bLResNeXt(nn.Module):

    def __init__(self, block, layers, basewidth, cardinality, alpha, beta, num_classes=1000):
        super(bLResNeXt, self
              ).__init__()
        num_channels = [64, 128, 256, 512]
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, num_channels[0], kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels[0])
        self.relu = nn.ReLU(inplace=True)

        self.b_conv0 = nn.Conv2d(num_channels[0], num_channels[0], kernel_size=3, stride=2, padding=1, bias=False)
        self.bn_b0 = nn.BatchNorm2d(num_channels[0])
        self.l_conv0 = nn.Conv2d(num_channels[0], num_channels[0] // alpha,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_l0 = nn.BatchNorm2d(num_channels[0] // alpha)
        self.l_conv1 = nn.Conv2d(num_channels[0] // alpha, num_channels[0] //
                                 alpha, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn_l1 = nn.BatchNorm2d(num_channels[0] // alpha)
        self.l_conv2 = nn.Conv2d(num_channels[0] // alpha, num_channels[0], kernel_size=1, stride=1, bias=False)
        self.bn_l2 = nn.BatchNorm2d(num_channels[0])

        self.bl_init = nn.Conv2d(num_channels[0], num_channels[0], kernel_size=1, stride=1, bias=False)
        self.bn_bl_init = nn.BatchNorm2d(num_channels[0])
        self.layer1 = bLModule(block, num_channels[0], num_channels[0] * block.expansion,
                               layers[0], basewidth, cardinality, alpha, beta, stride=2)
        self.layer2 = bLModule(block, num_channels[0] * block.expansion, num_channels[1]
                               * block.expansion, layers[1], basewidth, cardinality, alpha, beta, stride=2)
        self.layer3 = bLModule(block, num_channels[1] * block.expansion, num_channels[2]
                               * block.expansion, layers[2], basewidth, cardinality, alpha, beta, stride=1)
        self.layer4 = self._make_layer(
            block, num_channels[2] * block.expansion, num_channels[3] * block.expansion, layers[3], basewidth,
            cardinality, stride=2)
        self.gappool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(num_channels[3] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each block.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        for m in self.modules():
            if isinstance(m, Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)
        #     elif isinstance(m, BasicBlock):
        #         nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, inplanes, planes, blocks, basewidth, cardinality, stride=1):

        downsample = []
        if stride != 1:
            downsample.append(nn.AvgPool2d(3, stride=2, padding=1))
        if inplanes != planes:
            downsample.append(nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False))
            downsample.append(nn.BatchNorm2d(planes))
        downsample = None if downsample == [] else nn.Sequential(*downsample)

        layers = []
        layers.append(block(inplanes, planes, basewidth, cardinality, stride, downsample))
        for i in range(1, blocks):
            layers.append(block(planes, planes, basewidth, cardinality))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        bx = self.b_conv0(x)
        bx = self.bn_b0(bx)

        lx = self.l_conv0(x)
        lx = self.bn_l0(lx)
        lx = self.relu(lx)
        lx = self.l_conv1(lx)
        lx = self.bn_l1(lx)
        lx = self.relu(lx)
        lx = self.l_conv2(lx)
        lx = self.bn_l2(lx)

        x = self.relu(bx + lx)
        x = self.bl_init(x)
        x = self.bn_bl_init(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.gappool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def blresnext_model(depth, basewidth, cardinality, alpha, beta,
                    num_classes=1000, pretrained=False):
    layers = {
        50: [3, 4, 6, 3],
        101: [4, 8, 18, 3],
        152: [5, 12, 30, 3]
    }[depth]

    model = bLResNeXt(Bottleneck, layers, basewidth, cardinality,
                      alpha, beta, num_classes)
    if pretrained:
        url = model_urls['blresnext-{}-{}x{}d-a{}-b{}'.format(depth, cardinality,
                                                              basewidth, alpha, beta)]
        checkpoint = torch.load(url)
        model.load_state_dict(checkpoint['state_dict'])

    return model
