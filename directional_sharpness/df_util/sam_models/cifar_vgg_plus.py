# coding=utf-8
# Copyright 2026 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import torch.nn as nn
import torch.nn.init as init

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


class VGG(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, features,feature_size=512,num_classes=10):
        super(VGG, self).__init__()
        self.features = features
        self.num_classes = num_classes
        self.classifier = nn.Sequential(
            nn.Linear(feature_size, 128),
            nn.ReLU(True),
            nn.Linear(128,num_classes),
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, in_channels=3, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [16, 'M', 16, 'M', 32, 'M',  64, 'M', 64, 'M'],
    'A1': [16, 'M', 32, 'M', 32, 32, 'M', 64, 64, 'M', 128, 128, 'M'],
    'A2': [32, 'M', 64, 'M', 64, 64, 'M', 128, 128, 'M', 256, 256, 'M'],
    'A3': [64, 'M', 128, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M'],
    'A4': [128, 'M', 256, 'M', 256, 256, 'M', 512, 512, 'M', 1024, 1024, 'M'],
    'B': [16, 16, 'M', 32, 32, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
}


def vgg11(num_classes=10, in_channels=3):
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A'], in_channels),feature_size=64,num_classes=num_classes)


def vgg11_big(num_classes=10, in_channels=3):
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A3'],in_channels),cfg['A3'][-2],num_classes)

def vgg11_bn(num_classes, in_channels=3):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg['A'], in_channels, batch_norm=True), num_classes=num_classes)




def vgg13_mingze(num_classes=10, in_channels=3):
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B'], in_channels),cfg['B'][-2],num_classes=num_classes)




def vgg16_mingze(num_classes=10, in_channels=3):
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D'], in_channels),cfg['D'][-2],num_classes=num_classes)




def vgg19_mingze(num_classes=10, in_channels=3):
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E'], in_channels),cfg['E'][-2],num_classes=num_classes)

