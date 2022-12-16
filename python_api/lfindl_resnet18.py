from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy

class LFINDL_RESNET18:
    def __init__(self, model_path):

        self.data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }


        self.device = torch.device("cpu")

        self.model_conv = torch.load(model_path, map_location="cpu")
        self.model_conv = self.model_conv.to(self.device)

    def predict(self, image):

        with torch.no_grad():

            image = self.data_transforms['val'](image).unsqueeze(0).cpu()

            image = image.to(self.device)

            outputs = self.model_conv(image)
            _, preds = torch.max(outputs, 1)

            return preds
