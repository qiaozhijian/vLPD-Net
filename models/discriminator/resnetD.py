import MinkowskiEngine as ME
import numpy as np
import torch
import torch.nn as nn
from MinkowskiEngine.modules.resnet_block import BasicBlock

from misc.log import log_string
from models.discriminator.resnet import ResNetBase


class ResNet14(ResNetBase):
    BLOCK = BasicBlock
    LAYERS = (1, 1, 1, 1)
    INIT_DIM = 32
    PLANES = (32, 64, 128, 128)


class ResnetDor(torch.nn.Module):
    def __init__(self, planes=(32, 64, 64), n_layers=1, D=3):
        super().__init__()
        self.planes = planes
        self.n_layers = n_layers
        self.D = D
        self.network_initialization()

    def network_initialization(self):

        layers = len(self.planes)
        self.convs = nn.ModuleList()
        self.conv1 = nn.Sequential(
            ME.MinkowskiConvolution(self.planes[0], self.planes[0], kernel_size=2, stride=2, dimension=self.D),
            ME.MinkowskiBatchNorm(self.planes[0]),
            ME.MinkowskiReLU(inplace=True))
        self.convs.append(self.conv1)
        for i in range(layers - 1):
            self.convs.append(nn.Sequential(
                ME.MinkowskiConvolution(self.planes[i] * 2, self.planes[i + 1], kernel_size=2, stride=2,
                                        dimension=self.D),
                ME.MinkowskiBatchNorm(self.planes[i + 1]),
                ME.MinkowskiReLU(inplace=True)))

        self.pool = ME.MinkowskiGlobalAvgPooling()

        self.fc = []
        for j in range(self.n_layers):
            self.fc += [ME.MinkowskiLinear(self.planes[layers - 1], self.planes[layers - 1], bias=True)]
            self.fc += [ME.MinkowskiReLU(inplace=True)]
        self.fc += [ME.MinkowskiLinear(self.planes[layers - 1], 1, bias=True)]
        self.fc = nn.Sequential(*self.fc)

    # 32,32,64,64
    def forward(self, mid_feature):
        last_layer = None
        for i, conv in enumerate(self.convs):
            layer = mid_feature[i] if last_layer is None else ME.cat(last_layer, mid_feature[i])
            layer = conv(layer)
            last_layer = layer
        layer = self.pool(layer)
        pred = self.fc(layer)

        return pred.F

    def print_info(self):

        para = sum([np.prod(list(p.size())) for p in self.parameters()])
        log_string('Model {} : params: {:4f}M'.format(self._get_name(), para * 4 / 1000 / 1000))


class MeanFCDor(torch.nn.Module):
    def __init__(self, planes=(32, 64, 64), feature_dim=256, n_layers=3, D=3):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_layers = n_layers
        self.D = D
        self.network_initialization()
        self.pool = ME.MinkowskiGlobalAvgPooling()

    def network_initialization(self):
        self.fc = []
        lay_dim = self.feature_dim
        lay_dim = 960
        # for j in range(self.n_layers):
        self.fc += [nn.Linear(960, 64, bias=True)]
        self.fc += [nn.ReLU(inplace=True)]
        self.fc += [nn.Linear(64, 64, bias=True)]
        self.fc += [nn.ReLU(inplace=True)]
        self.fc += [nn.Linear(64, 1, bias=True)]
        self.fc = nn.Sequential(*self.fc)

    # 32,32,64,64
    def forward(self, mid_feature):
        mid_layers = []
        for i in range(len(mid_feature) - 1):
            mid_layers.append(self.pool(mid_feature[i]).F)
        mid_layers.append(mid_feature[-1])
        mid_layers = torch.cat(mid_layers, dim=1)
        pred = self.fc(mid_layers)

        return pred

    def print_info(self):
        para = sum([np.prod(list(p.size())) for p in self.parameters()])
        log_string('Model {} : params: {:4f}M'.format(self._get_name(), para * 4 / 1000 / 1000))


class FCDor(torch.nn.Module):
    def __init__(self, feature_dim=256, n_layers=3, D=3):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_layers = n_layers
        self.D = D
        self.network_initialization()

    def network_initialization(self):
        self.fc = []
        lay_dim = self.feature_dim
        for j in range(self.n_layers):
            self.fc += [nn.Linear(lay_dim, lay_dim // (j + 1), bias=True)]
            self.fc += [nn.ReLU(inplace=True)]
            lay_dim = lay_dim // (j + 1)
        self.fc += [nn.Linear(lay_dim, 1, bias=True)]
        self.fc = nn.Sequential(*self.fc)

    # 32,32,64,64
    def forward(self, mid_feature):
        # layer = mid_feature
        pred = self.fc(mid_feature)

        return pred

    def print_info(self):
        para = sum([np.prod(list(p.size())) for p in self.parameters()])
        log_string('Model {} : params: {:4f}M'.format(self._get_name(), para * 4 / 1000 / 1000))


class ResnetFDor(torch.nn.Module):
    def __init__(self, planes=(32, 64, 64), n_layers=1, D=3):
        super().__init__()
        self.planes = planes
        self.n_layers = n_layers
        self.D = D
        self.network_initialization()

    def network_initialization(self):

        layers = len(self.planes)
        self.convs = nn.ModuleList()
        self.conv1 = nn.Sequential(
            nn.Conv1d(self.planes[0], self.planes[0], kernel_size=2, stride=2),
            nn.BatchNorm1d(self.planes[0]),
            nn.ReLU(inplace=True))
        self.convs.append(self.conv1)
        for i in range(layers - 1):
            self.convs.append(nn.Sequential(
                nn.Conv1d(self.planes[i] * 2, self.planes[i + 1], kernel_size=2, stride=2),
                nn.BatchNorm1d(self.planes[i + 1]),
                nn.ReLU(inplace=True)))

        self.pool = lambda x: torch.mean(x, dim=-1, keepdim=True)

        self.fc = []
        for j in range(self.n_layers):
            self.fc += [nn.Linear(self.planes[layers - 1], self.planes[layers - 1], bias=True)]
            self.fc += [nn.ReLU(inplace=True)]
        self.fc += [nn.Linear(self.planes[layers - 1], 1, bias=True)]
        self.fc = nn.Sequential(*self.fc)
        pass

    # 32,32,64,64
    def forward(self, mid_feature):
        last_layer = None
        for i, conv in enumerate(self.convs):
            layer = mid_feature[i] if last_layer is None else torch.cat([last_layer, mid_feature[i]], dim=2)
            layer = conv(layer)
            last_layer = layer
        layer = self.pool(layer)
        pred = self.fc(layer)

        return pred

    def print_info(self):

        para = sum([np.prod(list(p.size())) for p in self.parameters()])
        log_string('Model {} : params: {:4f}M'.format(self._get_name(), para * 4 / 1000 / 1000))
