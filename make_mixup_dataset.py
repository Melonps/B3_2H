import argparse
import math
import os
import pickle
import warnings
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import cuda, optim
from torch.utils import data
#from torchvision import datasets, transforms
from torchvision import datasets, transforms
from tqdm import tqdm

from generative_model import GenerativeModel
from loss import One_Hot_CrossEntropy

warnings.simplefilter(action="ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--start', type=int)
parser.add_argument('--end', type=int)

args = parser.parse_args()

if __name__ == "__main__":
    weight_path = "weight/weight1.pth"
    save_data_folder = "../data/cifar10-modified/"
    

    if not os.path.exists(save_data_folder):
        os.makedirs(save_data_folder)

    transform = transforms.Compose([transforms.ToTensor()])
    trainloader = data.DataLoader(
        dataset=datasets.CIFAR10(root="../data", train=True, transform=transform),
        batch_size=1,
        shuffle=False,
    )

    generative_num = args.end - args.start

    sub_trainloader = data.DataLoader(
        dataset=datasets.CIFAR10(root="../data", train=True, transform=transform),
        batch_size=generative_num,
        shuffle=True,
    )
    lr = 0.05
    momentum = 0.9
    weight_decay = 5e-4
    eps = 10e-8
    
    #print(generative_num)

    criterion = One_Hot_CrossEntropy()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    #print(device)
    
    input = torch.tensor([[1.0]], requires_grad=True)
    input = input.to(device=device, non_blocking=True)

    ideal_targets = []
    generated_images = []
    output_targets = []
    
    for k, (sub_image, sub_target) in enumerate(tqdm(sub_trainloader)):
        break
    sub_image = sub_image.cuda(non_blocking=True)
    sub_target = sub_target.cuda(non_blocking=True)
    sub_target = F.one_hot(sub_target, num_classes=10)

    #print(sub_target[1])
    #print(sub_target[2])
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    j=0

    for i, (image, target) in enumerate(tqdm(trainloader)):
        if i < args.start:
            continue
        if i >= args.end:
            break

        image = image.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        ld = np.random.beta(1,1,1)
        ld = torch.from_numpy(ld.astype(np.float32)).clone()
        #print(ld)
        ld = ld.to('cuda')
        
        # 理想の出力の設定
        target = F.one_hot(target, num_classes=10)
        #print(target)
        target = target * ld + sub_target[j] * (1 - ld)
        ideal_targets.append(target)
        output_targets.append(target)
        print(target)

        #print(ld)
        #print(1-ld)

        #画像生成
        generated_image = (image * ld + sub_image[j] * (1-ld))
        generated_npimg = generated_image.to('cpu').detach().numpy().copy()
        generated_images.append(generated_npimg)
        j = j + 1
    

    with open(save_data_folder + 'ideal_target{}_{}'.format(args.start, args.end - 1), 'wb') as f:
        pickle.dump(ideal_targets, f)

    with open(save_data_folder + 'output_target{}_{}'.format(args.start, args.end - 1), 'wb') as f:
        pickle.dump(output_targets, f)

    with open(save_data_folder + 'generated_image{}_{}'.format(args.start, args.end - 1), 'wb') as f:
        pickle.dump(generated_images, f)
    