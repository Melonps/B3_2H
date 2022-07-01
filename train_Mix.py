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

    model.classifier.load_state_dict(state_dict)
    model.eval()

    

    criterion1 = One_Hot_CrossEntropy()
    criterion1 = criterion1.cuda()
    criterion2 = torch.nn.CrossEntropyLoss()
    criterion2 = criterion2.cuda()

    classifier_optimizer = optim.SGD(model.classifier.parameters(), lr=0.001, momentum=momentum, weight_decay=weight_decay)

    input = torch.tensor([[1.0]], requires_grad=True)
    input = input.cuda(non_blocking=True)

    for epoch in range(1):
        for i, (image, target) in enumerate(tqdm.tqdm(trainloader)):
            image = image.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            print(target.dtype)

            for k, (sub_image, sub_target) in enumerate(tqdm.tqdm(trainloader)):
                break
            sub_image = sub_image.cuda(non_blocking=True)
            sub_target = sub_target.cuda(non_blocking=True)

            #print(sub_target)


            for j in range(generative_num):

                ld = np.random.beta(1,1,1)
                ld = torch.from_numpy(ld.astype(np.float32)).clone()
                #print(ld)
                #print((ld.device))
                ld = ld.to('cuda')

                #理想の出力の設定
                ideal_target_1 = F.one_hot(target[j], num_classes=10)
                ideal_target_2 = F.one_hot(sub_target[generative_num + j], num_classes=10) #generative_numの数だけずらして使う

                #print((ideal_target_1.device))

                ideal_target = ideal_target_1 * ld + ideal_target_2 * (1-ld) #ラベルの合成(今は0.5 0.5 )
                print(ideal_target.dtype)

                #画像を重みとして設定
                image_as_weight = torch.clamp(image[j], min=eps, max=1.0-eps)
                image_as_weight = torch.tan((image_as_weight - 0.5)*math.pi)
                model.additional_layer.weight.data = image_as_weight.view(-1, 1)


                #画像生成
                generated_image = (image[j] * ld + sub_image[generative_num + j] * (1-ld))
                image[j] = generated_image/math.pi + 0.5
                #画像の保存
                #fig_save_path = fig_save_folder + 'original_{}.png'.format(i)
                #generated_image = generated_image.cpu().numpy()
                #plt.imshow(np.transpose(generated_image, (1, 2, 0)))
                #plt.savefig(fig_save_path)
                #plt.close()


            #分類モデルの学習
            model.train()

            output = model(image)
            if torch.isnan(output).any():
                print('nan is detected')
                exit(1)
            print(ideal_target)
            loss = criterion1(output, ideal_target)

            classifier_optimizer.zero_grad()
            loss.backward()
            classifier_optimizer.step()

            print('\nepoch:{:2d} iter:{:3d}/{} loss={:.4f}'.format(epoch+1, i+1, len(trainloader), loss.data), end='\r\033[1A')

        torch.save(model.classifier.state_dict(), 'weight_of_generative_training/checkpoint_{}.pth'.format(epoch))