from collections import namedtuple

import torch
from torchvision import models


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.relu1_1 = torch.nn.Sequential()
        self.relu2_1 = torch.nn.Sequential()
        self.relu3_1 = torch.nn.Sequential()
        self.relu4_1 = torch.nn.Sequential()
        self.relu5_1 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), vgg_pretrained_features[x])

        for x in range(2,7):
            self.relu2_1.add_module(str(x), vgg_pretrained_features[x])

        for x in range(7,12):
            self.relu3_1.add_module(str(x), vgg_pretrained_features[x])

        for x in range(12,21):
            self.relu4_1.add_module(str(x), vgg_pretrained_features[x])

        for x in range(21,30):
            self.relu5_1.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.relu1_1(X)
        h_relu1_1 = h
        h = self.relu2_1(h)
        h_relu2_1 = h
        h = self.relu3_1(h)
        h_relu3_1 = h
        h = self.relu4_1(h)
        h_relu4_1 = h
        h = self.relu5_1(h)
        h_relu5_1 = h
        return [h_relu1_1, h_relu2_1, h_relu3_1, h_relu4_1, h_relu5_1]