import math

import torch
from torch import nn
from torchvision import transforms

from vgg import vgg16

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class GenerativeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.additional_layer = nn.Linear(1, 3 * 32 * 32, bias=False)
        self.classifier = vgg16()

    def generative(self, x):
        x = self.additional_layer(x)
        x = x.view(1, 3, 32, 32)
        x = torch.arctan(x)
        x = x / math.pi + 0.5
        x = normalize(x)
        x = self.classifier(x)
        return x

    def forward(self, x):
        x = normalize(x)
        x = self.classifier(x)
        return x


def generative_model():
    return GenerativeModel()
