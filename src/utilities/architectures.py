from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

"""
LENET 5
"""


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1, bias=True)
        self.r1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1, bias=True)
        self.r2 = nn.ReLU()
        self.fc1 = nn.Linear(4 * 4 * 50, 500, bias=True)
        self.r3 = nn.ReLU()
        self.fc2 = nn.Linear(500, 10, bias=True)
    
    def forward(self, img):
        output = self.conv1(img)
        output = self.r1(output)
        output = F.max_pool2d(output, 2)
        output = self.conv2(output)
        output = self.r2(output)
        output = F.max_pool2d(output, 2)
        output = output.view(img.size(0), -1)
        output = self.fc1(output)
        output = self.r3(output)
        output = self.fc2(output)
        
        return output


"""
LENET 300
"""


class LeNet300(nn.Module):
    def __init__(self):
        super(LeNet300, self).__init__()
        self.fc1 = nn.Linear(784, 300, bias=True)
        self.r1 = nn.ReLU()
        self.fc2 = nn.Linear(300, 100, bias=True)
        self.r2 = nn.ReLU()
        self.fc3 = nn.Linear(100, 10, bias=True)
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = self.r1(x)
        x = self.fc2(x)
        x = self.r2(x)
        x = self.fc3(x)
        return x


"""
MOBILENET V2
"""


class MobilenetBlock(nn.Module):
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(MobilenetBlock, self).__init__()
        self.stride = stride
        
        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
        
        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )
    
    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = F.relu(out)
        out = self.bn2(self.conv2(out))
        out = F.relu(out)
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1, 16, 1, 1),
           (6, 24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6, 32, 3, 2),
           (6, 64, 4, 2),
           (6, 96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]
    
    def __init__(self, num_classes=10):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)
    
    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1] * (num_blocks - 1)
            for stride in strides:
                layers.append(MobilenetBlock(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = F.relu(out)
        for lay in self.layers:
            out = lay(out)
        out = self.bn2(self.conv2(out))
        out = F.relu(out)
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


"""
RESNET CIFAR10
"""


class ResnetLambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(ResnetLambdaLayer, self).__init__()
        self.lambd = lambd
    
    def forward(self, x):
        return self.lambd(x)


class ResnetBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(ResnetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = None
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                Increses dimension via padding, performs identity operations
                """
                self.shortcut = ResnetLambdaLayer(lambda x:
                                                  F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4),
                                                        "constant",
                                                        0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )
        
        self.relu2 = nn.ReLU(inplace=False)
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.shortcut is not None:
            identity = self.shortcut(x)
        
        out += identity
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, option="A"):
        super(ResNet, self).__init__()
        self.in_planes = 16
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=False)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, option=option)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, option=option)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, option=option)
        self.linear = nn.Linear(64, num_classes)
    
    # self.apply(_weights_init)
    
    def _make_layer(self, block, planes, num_blocks, stride, option):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, option))
            self.in_planes = planes * block.expansion
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        pool_size = int(out.size(3))
        out = F.avg_pool2d(out, pool_size)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet32(option="A"):
    return ResNet(ResnetBlock, [5, 5, 5], option=option)


"""
VGG CIFAR10 1
"""
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG1L(nn.Module):
    def __init__(self, vgg_name="VGG16"):
        super(VGG1L, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)
    
    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
    
    @staticmethod
    def _make_layers(cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


"""
VGG CIFAR10 2
"""


class VGG2L(nn.Module):
    def __init__(self, classes=10):
        super(VGG2L, self).__init__()
        self.features = self._make_layers()
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    @staticmethod
    def _make_layers():
        layers = []
        layers += [nn.Conv2d(3, 64, kernel_size=3, padding=1)]
        layers += [nn.BatchNorm2d(64, eps=1e-3)]
        layers += [nn.ReLU()]
        layers += [nn.Dropout(0.3)]
        
        layers += [nn.Conv2d(64, 64, kernel_size=3, padding=1)]
        layers += [nn.BatchNorm2d(64, eps=1e-3)]
        layers += [nn.ReLU()]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        
        layers += [nn.Conv2d(64, 128, kernel_size=3, padding=1)]
        layers += [nn.BatchNorm2d(128, eps=1e-3)]
        layers += [nn.ReLU()]
        layers += [nn.Dropout(0.4)]
        
        layers += [nn.Conv2d(128, 128, kernel_size=3, padding=1)]
        layers += [nn.BatchNorm2d(128, eps=1e-3)]
        layers += [nn.ReLU()]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        
        layers += [nn.Conv2d(128, 256, kernel_size=3, padding=1)]
        layers += [nn.BatchNorm2d(256, eps=1e-3)]
        layers += [nn.ReLU()]
        layers += [nn.Dropout(0.4)]
        
        layers += [nn.Conv2d(256, 256, kernel_size=3, padding=1)]
        layers += [nn.BatchNorm2d(256, eps=1e-3)]
        layers += [nn.ReLU()]
        layers += [nn.Dropout(0.4)]
        
        layers += [nn.Conv2d(256, 256, kernel_size=3, padding=1)]
        layers += [nn.BatchNorm2d(256, eps=1e-3)]
        layers += [nn.ReLU()]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        
        layers += [nn.Conv2d(256, 512, kernel_size=3, padding=1)]
        layers += [nn.BatchNorm2d(512, eps=1e-3)]
        layers += [nn.ReLU()]
        layers += [nn.Dropout(0.4)]
        
        layers += [nn.Conv2d(512, 512, kernel_size=3, padding=1)]
        layers += [nn.BatchNorm2d(512, eps=1e-3)]
        layers += [nn.ReLU()]
        layers += [nn.Dropout(0.4)]
        
        layers += [nn.Conv2d(512, 512, kernel_size=3, padding=1)]
        layers += [nn.BatchNorm2d(512, eps=1e-3)]
        layers += [nn.ReLU()]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        
        layers += [nn.Conv2d(512, 512, kernel_size=3, padding=1)]
        layers += [nn.BatchNorm2d(512, eps=1e-3)]
        layers += [nn.ReLU()]
        layers += [nn.Dropout(0.4)]
        
        layers += [nn.Conv2d(512, 512, kernel_size=3, padding=1)]
        layers += [nn.BatchNorm2d(512, eps=1e-3)]
        layers += [nn.ReLU()]
        layers += [nn.Dropout(0.4)]
        
        layers += [nn.Conv2d(512, 512, kernel_size=3, padding=1)]
        layers += [nn.BatchNorm2d(512, eps=1e-3)]
        layers += [nn.ReLU()]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        return nn.Sequential(*layers)


"""
ALEXNET CIFAR100
"""


class AlexNet(nn.Module):
    def __init__(self, num_classes=100):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes, bias=False),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x


"""
SEGENET
"""

DEBUG = False

vgg16_dims = [
    (64, 64, 'M'),  # Stage - 1
    (128, 128, 'M'),  # Stage - 2
    (256, 256, 256, 'M'),  # Stage - 3
    (512, 512, 512, 'M'),  # Stage - 4
    (512, 512, 512, 'M')  # Stage - 5
]

decoder_dims = [
    ('U', 512, 512, 512),  # Stage - 5
    ('U', 512, 512, 512),  # Stage - 4
    ('U', 256, 256, 256),  # Stage - 3
    ('U', 128, 128),  # Stage - 2
    ('U', 64, 64)  # Stage - 1
]


class SegNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(SegNet, self).__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        self.num_channels = input_channels
        
        # Encoder layers
        
        self.encoder_conv_00 = nn.Sequential(*[
            nn.Conv2d(in_channels=self.input_channels,
                      out_channels=64,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        ])
        self.encoder_conv_01 = nn.Sequential(*[
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        ])
        self.encoder_conv_10 = nn.Sequential(*[
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        ])
        self.encoder_conv_11 = nn.Sequential(*[
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        ])
        self.encoder_conv_20 = nn.Sequential(*[
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        ])
        self.encoder_conv_21 = nn.Sequential(*[
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        ])
        self.encoder_conv_22 = nn.Sequential(*[
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        ])
        self.encoder_conv_30 = nn.Sequential(*[
            nn.Conv2d(in_channels=256,
                      out_channels=512,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        ])
        self.encoder_conv_31 = nn.Sequential(*[
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        ])
        self.encoder_conv_32 = nn.Sequential(*[
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        ])
        self.encoder_conv_40 = nn.Sequential(*[
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        ])
        self.encoder_conv_41 = nn.Sequential(*[
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        ])
        self.encoder_conv_42 = nn.Sequential(*[
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        ])
        
        self.init_vgg_weights()
        
        # Decoder layers
        
        self.decoder_convtr_42 = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=512,
                               out_channels=512,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        ])
        self.decoder_convtr_41 = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=512,
                               out_channels=512,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        ])
        self.decoder_convtr_40 = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=512,
                               out_channels=512,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        ])
        self.decoder_convtr_32 = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=512,
                               out_channels=512,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        ])
        self.decoder_convtr_31 = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=512,
                               out_channels=512,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        ])
        self.decoder_convtr_30 = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=512,
                               out_channels=256,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        ])
        self.decoder_convtr_22 = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=256,
                               out_channels=256,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        ])
        self.decoder_convtr_21 = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=256,
                               out_channels=256,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        ])
        self.decoder_convtr_20 = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=256,
                               out_channels=128,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        ])
        self.decoder_convtr_11 = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=128,
                               out_channels=128,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        ])
        self.decoder_convtr_10 = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=128,
                               out_channels=64,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        ])
        self.decoder_convtr_01 = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=64,
                               out_channels=64,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        ])
        self.decoder_convtr_00 = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=64,
                               out_channels=self.output_channels,
                               kernel_size=3,
                               padding=1)
        ])
    
    def forward(self, input_img):
        """
        Forward pass `input_img` through the network
        """
        
        # Encoder
        
        # Encoder Stage - 1
        dim_0 = input_img.size()
        x_00 = self.encoder_conv_00(input_img)
        x_01 = self.encoder_conv_01(x_00)
        x_0, indices_0 = F.max_pool2d(x_01, kernel_size=2, stride=2, return_indices=True)
        
        # Encoder Stage - 2
        dim_1 = x_0.size()
        x_10 = self.encoder_conv_10(x_0)
        x_11 = self.encoder_conv_11(x_10)
        x_1, indices_1 = F.max_pool2d(x_11, kernel_size=2, stride=2, return_indices=True)
        
        # Encoder Stage - 3
        dim_2 = x_1.size()
        x_20 = self.encoder_conv_20(x_1)
        x_21 = self.encoder_conv_21(x_20)
        x_22 = self.encoder_conv_22(x_21)
        x_2, indices_2 = F.max_pool2d(x_22, kernel_size=2, stride=2, return_indices=True)
        
        # Encoder Stage - 4
        dim_3 = x_2.size()
        x_30 = self.encoder_conv_30(x_2)
        x_31 = self.encoder_conv_31(x_30)
        x_32 = self.encoder_conv_32(x_31)
        x_3, indices_3 = F.max_pool2d(x_32, kernel_size=2, stride=2, return_indices=True)
        
        # Encoder Stage - 5
        dim_4 = x_3.size()
        x_40 = self.encoder_conv_40(x_3)
        x_41 = self.encoder_conv_41(x_40)
        x_42 = self.encoder_conv_42(x_41)
        x_4, indices_4 = F.max_pool2d(x_42, kernel_size=2, stride=2, return_indices=True)
        
        # Decoder
        
        # Decoder Stage - 5
        x_4d = F.interpolate(x_4, scale_factor=2)
        x_42d = self.decoder_convtr_42(x_4d)
        x_41d = self.decoder_convtr_41(x_42d)
        x_40d = self.decoder_convtr_40(x_41d)
        
        # Decoder Stage - 4
        x_3d = F.interpolate(x_40d, scale_factor=2)
        x_32d = self.decoder_convtr_32(x_3d)
        x_31d = self.decoder_convtr_31(x_32d)
        x_30d = self.decoder_convtr_30(x_31d)
        
        # Decoder Stage - 3
        x_2d = F.interpolate(x_30d, scale_factor=2)
        x_22d = self.decoder_convtr_22(x_2d)
        x_21d = self.decoder_convtr_21(x_22d)
        x_20d = self.decoder_convtr_20(x_21d)
        
        # Decoder Stage - 2
        x_1d = F.interpolate(x_20d, scale_factor=2)
        x_11d = self.decoder_convtr_11(x_1d)
        x_10d = self.decoder_convtr_10(x_11d)
        
        # Decoder Stage - 1
        x_0d = F.interpolate(x_10d, scale_factor=2)
        x_01d = self.decoder_convtr_01(x_0d)
        x_00d = self.decoder_convtr_00(x_01d)
        
        x_softmax = F.softmax(x_00d, dim=1)
        
        return x_00d
    
    def init_vgg_weights(self):
        vgg16 = models.vgg16(pretrained=True)
        
        assert self.encoder_conv_00[0].weight.size() == vgg16.features[0].weight.size()
        self.encoder_conv_00[0].weight.data = vgg16.features[0].weight.data
        assert self.encoder_conv_00[0].bias.size() == vgg16.features[0].bias.size()
        self.encoder_conv_00[0].bias.data = vgg16.features[0].bias.data
        
        assert self.encoder_conv_01[0].weight.size() == vgg16.features[2].weight.size()
        self.encoder_conv_01[0].weight.data = vgg16.features[2].weight.data
        assert self.encoder_conv_01[0].bias.size() == vgg16.features[2].bias.size()
        self.encoder_conv_01[0].bias.data = vgg16.features[2].bias.data
        
        assert self.encoder_conv_10[0].weight.size() == vgg16.features[5].weight.size()
        self.encoder_conv_10[0].weight.data = vgg16.features[5].weight.data
        assert self.encoder_conv_10[0].bias.size() == vgg16.features[5].bias.size()
        self.encoder_conv_10[0].bias.data = vgg16.features[5].bias.data
        
        assert self.encoder_conv_11[0].weight.size() == vgg16.features[7].weight.size()
        self.encoder_conv_11[0].weight.data = vgg16.features[7].weight.data
        assert self.encoder_conv_11[0].bias.size() == vgg16.features[7].bias.size()
        self.encoder_conv_11[0].bias.data = vgg16.features[7].bias.data
        
        assert self.encoder_conv_20[0].weight.size() == vgg16.features[10].weight.size()
        self.encoder_conv_20[0].weight.data = vgg16.features[10].weight.data
        assert self.encoder_conv_20[0].bias.size() == vgg16.features[10].bias.size()
        self.encoder_conv_20[0].bias.data = vgg16.features[10].bias.data
        
        assert self.encoder_conv_21[0].weight.size() == vgg16.features[12].weight.size()
        self.encoder_conv_21[0].weight.data = vgg16.features[12].weight.data
        assert self.encoder_conv_21[0].bias.size() == vgg16.features[12].bias.size()
        self.encoder_conv_21[0].bias.data = vgg16.features[12].bias.data
        
        assert self.encoder_conv_22[0].weight.size() == vgg16.features[14].weight.size()
        self.encoder_conv_22[0].weight.data = vgg16.features[14].weight.data
        assert self.encoder_conv_22[0].bias.size() == vgg16.features[14].bias.size()
        self.encoder_conv_22[0].bias.data = vgg16.features[14].bias.data
        
        assert self.encoder_conv_30[0].weight.size() == vgg16.features[17].weight.size()
        self.encoder_conv_30[0].weight.data = vgg16.features[17].weight.data
        assert self.encoder_conv_30[0].bias.size() == vgg16.features[17].bias.size()
        self.encoder_conv_30[0].bias.data = vgg16.features[17].bias.data
        
        assert self.encoder_conv_31[0].weight.size() == vgg16.features[19].weight.size()
        self.encoder_conv_31[0].weight.data = vgg16.features[19].weight.data
        assert self.encoder_conv_31[0].bias.size() == vgg16.features[19].bias.size()
        self.encoder_conv_31[0].bias.data = vgg16.features[19].bias.data
        
        assert self.encoder_conv_32[0].weight.size() == vgg16.features[21].weight.size()
        self.encoder_conv_32[0].weight.data = vgg16.features[21].weight.data
        assert self.encoder_conv_32[0].bias.size() == vgg16.features[21].bias.size()
        self.encoder_conv_32[0].bias.data = vgg16.features[21].bias.data
        
        assert self.encoder_conv_40[0].weight.size() == vgg16.features[24].weight.size()
        self.encoder_conv_40[0].weight.data = vgg16.features[24].weight.data
        assert self.encoder_conv_40[0].bias.size() == vgg16.features[24].bias.size()
        self.encoder_conv_40[0].bias.data = vgg16.features[24].bias.data
        
        assert self.encoder_conv_41[0].weight.size() == vgg16.features[26].weight.size()
        self.encoder_conv_41[0].weight.data = vgg16.features[26].weight.data
        assert self.encoder_conv_41[0].bias.size() == vgg16.features[26].bias.size()
        self.encoder_conv_41[0].bias.data = vgg16.features[26].bias.data
        
        assert self.encoder_conv_42[0].weight.size() == vgg16.features[28].weight.size()
        self.encoder_conv_42[0].weight.data = vgg16.features[28].weight.data
        assert self.encoder_conv_42[0].bias.size() == vgg16.features[28].bias.size()
        self.encoder_conv_42[0].bias.data = vgg16.features[28].bias.data
        
        del vgg16


"""
UNET
"""


class UNet(nn.Module):
    
    def __init__(self, in_channels=1, out_channels=1, init_features=64):
        super(UNet, self).__init__()
        
        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.encoder5 = UNet._block(features * 8, features * 16, name="enc5")
        self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.bottleneck = UNet._block(features * 16, features * 32, name="bottleneck")
        
        self.upconv5 = nn.ConvTranspose2d(
            features * 32, features * 16, kernel_size=2, stride=2
        )
        self.decoder5 = UNet._block((features * 16) * 2, features * 16, name="dec5")
        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")
        
        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )
    
    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        enc5 = self.encoder5(self.pool4(enc4))
        
        bottleneck = self.bottleneck(self.pool5(enc5))
        
        dec5 = self.upconv5(bottleneck)
        dec5 = torch.cat((dec5, enc5), dim=1)
        dec5 = self.decoder5(dec5)
        dec4 = self.upconv4(dec5)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return self.conv(dec1)
    
    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
