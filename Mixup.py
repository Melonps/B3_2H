import math
import random

import torch
import torch.nn
import torch.nn.functional as F
import tqdm
import numpy as np

from matplotlib import pyplot as plt
from torch import optim
from torchvision import datasets, transforms

from generative_model import generative_model
from loss import One_Hot_CrossEntropy

if __name__ == '__main__':
    weight_path = 'weight/checkpoint_299_without_ddrop.pth'

    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.CIFAR10(root='../data', train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        dataset=trainset,
        batch_size=128,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    lr = 0.05
    momentum = 0.9
    weight_decay = 5e-4
    eps = 10e-8
    generative_num = 8
    generative_backprop_num = 100

    fig_save_folder = 'Mixed_img/'

    model = generative_model()
    model.cuda()

    try:
        state_dict = torch.load(weight_path)
    except FileNotFoundError:
        print('Weights not found.')

    #model.classifier.load_state_dict(state_dict)
    model.eval()

    criterion1 = One_Hot_CrossEntropy()
    criterion1 = criterion1.cuda()
    criterion2 = torch.nn.CrossEntropyLoss()
    criterion2 = criterion2.cuda()

    classifier_optimizer = optim.SGD(model.classifier.parameters(), lr=0.001, momentum=momentum, weight_decay=weight_decay)

    input = torch.tensor([[1.0]], requires_grad=True)
    input = input.cuda(non_blocking=True)

    for epoch in range(300):
        for i, (image, target) in enumerate(tqdm.tqdm(trainloader)):
            image = image.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            

            #分類モデルの学習
            model.train()

            output = model(image)
            if torch.isnan(output).any():
                print('nan is detected')
                exit(1)
            loss = criterion2(output, target)

            classifier_optimizer.zero_grad()
            loss.backward()
            classifier_optimizer.step()

            print('\nepoch:{:2d} iter:{:3d}/{} loss={:.4f}'.format(epoch+1, i+1, len(trainloader), loss.data), end='\r\033[1A')
        
    torch.save(model.classifier.state_dict(), 'weight_of_generative_training/checkpoint_{}.pth'.format(epoch))