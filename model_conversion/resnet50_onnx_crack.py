#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import cv2

from PIL import Image

cudnn.benchmark = True
plt.ion()   # interactive mode


# In[2]:


data_transforms = {
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_conv = torch.load("../train/resnet50_conv_crack.pth")

model_conv = model_conv.to(device)

model_conv.eval()

test_image = '../train/data/crack_detection/test/crack/18000_1.jpg'

image = Image.open(test_image)

numpy_image=np.array(image)

image = data_transforms['val'](image).unsqueeze(0).cuda()

image = image.to(device)

outputs = model_conv(image)

# 모델 변환
torch.onnx.export(model_conv,               # 실행될 모델
                  image,                         # 모델 입력값 (튜플 또는 여러 입력값들도 가능)
                  "resnet50_conv_crack.onnx",   # 모델 저장 경로 (파일 또는 파일과 유사한 객체 모두 가능)
                  export_params=True,        # 모델 파일 안에 학습된 모델 가중치를 저장할지의 여부
                  opset_version=10,          # 모델을 변환할 때 사용할 ONNX 버전
                  do_constant_folding=True,  # 최적화시 상수폴딩을 사용할지의 여부
                  input_names = ['input'],   # 모델의 입력값을 가리키는 이름
                  output_names = ['output'], # 모델의 출력값을 가리키는 이름
                  dynamic_axes={'input' : {0 : 'batch_size'},    # 가변적인 길이를 가진 차원
                                'output' : {0 : 'batch_size'}})

print('model conversion compilete !!!')