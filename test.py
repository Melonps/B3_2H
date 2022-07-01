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
        batch_size=10,
        shuffle=True,
        num_workers=2,
        pin_memory=False
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

    model.classifier.load_state_dict(state_dict)
    model.eval()

    ld = np.random.beta(1,1,1)
    ld = torch.from_numpy(ld.astype(np.float32)).clone()
    #print(ld)
    #print((ld.device))
    ld = ld.to('cuda')

    criterion1 = One_Hot_CrossEntropy()
    criterion1 = criterion1.cuda()
    criterion2 = torch.nn.CrossEntropyLoss()
    criterion2 = criterion2.cuda()

    classifier_optimizer = optim.SGD(model.classifier.parameters(), lr=0.001, momentum=momentum, weight_decay=weight_decay)

    input = torch.tensor([[1.0]], requires_grad=True)
    input = input.cuda(non_blocking=True)

    
    for k, (sub_image, sub_target) in enumerate(tqdm.tqdm(trainloader)):
            break
    sub_image = sub_image.cuda(non_blocking=True)
    sub_target = sub_target.cuda(non_blocking=True)

    print(sub_target[1])
    print(1-ld)
    print(sub_target[1]*(1-ld))