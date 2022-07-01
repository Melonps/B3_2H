import math
import random

import torch
import torch.nn
import torch.nn.functional as F
import tqdm
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
    generative_num = 32
    generative_backprop_num = 100


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

    for epoch in range(50):
        for i, (image, target) in enumerate(tqdm.tqdm(trainloader)):
            image = image.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            for j in range(generative_num):
                #理想の出力の設定
                ideal_target = F.one_hot(target[j], num_classes=10)
                ideal_target = ideal_target / 2 + F.one_hot(torch.tensor(random.randrange(10)).cuda(non_blocking=True), num_classes=10) / 2

                #画像を重みとして設定
                image_as_weight = torch.clamp(image[j], min=eps, max=1.0-eps)
                image_as_weight = torch.tan((image_as_weight - 0.5)*math.pi)
                model.additional_layer.weight.data = image_as_weight.view(-1, 1)

                #画像生成
                additional_layer_optimizer = optim.SGD(model.additional_layer.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
                for k in range(generative_backprop_num):
                    gene_output = model.generative(input)
                    gene_loss = criterion1(gene_output, ideal_target)

                    additional_layer_optimizer.zero_grad()
                    gene_loss.backward()
                    additional_layer_optimizer.step()

                #画像の取り出し
                generated_image = model.additional_layer.weight.data.view(3, 32, 32).detach().clone()
                generated_image = torch.arctan(generated_image)
                image[j] = generated_image/math.pi + 0.5


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
