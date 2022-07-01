import math
import torch
from torch import optim
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from generative_model import generative_model
from loss import One_Hot_CrossEntropy


if __name__ == '__main__':
    fig_save_folder = 'generated_img/'

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    transform = transforms.Compose([transforms.ToTensor()])
    testset = datasets.CIFAR10(root='../data', train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(
        dataset=testset,
        batch_size=1,
        shuffle=False
    )

    lr = 0.05
    momentum = 0.9
    weight_decay = 5e-4
    eps = 10e-8

    model = generative_model()
    model.cuda()

    batch_sizes = [8, 16, 32]

    for b in batch_sizes:
        for e in range(10):
            weight_path = 'weight_of_generative_training/checkpoint_{}_b{:02d}.pth'.format(e, b)
            try:
                state_dict = torch.load(weight_path)
            except FileNotFoundError:
                print('Weights not found.')

            model.classifier.load_state_dict(state_dict)
            model.eval()

            criterion = One_Hot_CrossEntropy()
            criterion = criterion.cuda()

            #生成する画像の枚数と重み更新の回数
            generate_num = 10
            backprop_num = 100

            input = torch.tensor([[1.0]], requires_grad=True)
            input = input.cuda(non_blocking=True)

            for i, (image, target) in enumerate(testloader):
                if i >= generate_num:
                    break

                #理想の出力の設定
                # target = F.one_hot(target, num_classes=10)
                target = torch.tensor([[0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0]])
                target = target.cuda(non_blocking=True)

                #画像を重みとして設定
                image = image.cuda(non_blocking=True)
                image = torch.clamp(image, min=eps, max=1.0-eps)
                image = torch.tan((image - 0.5)*math.pi)
                model.additional_layer.weight.data = image.view(-1, 1)

                losses = []
                optimizer = optim.SGD(model.additional_layer.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

                #重み更新
                for j in range(backprop_num):
                    output = model.generative(input)
                    loss = criterion(output, target)

                    losses.append(loss.cpu().detach().numpy())

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                #生成画像の保存
                generated_image = model.additional_layer.weight.data.view(3, 32, 32).clone().cpu()
                generated_image = torch.arctan(generated_image)
                generated_image = generated_image/math.pi + 0.5
                generated_npimg = generated_image.numpy()
                fig_save_path = fig_save_folder + 'e{}_b{:02d}_{}.png'.format(e, b, i)
                plt.imshow(np.transpose(generated_npimg, (1, 2, 0)))
                plt.title('epoch {}'.format(e))
                plt.savefig(fig_save_path)
                plt.close()
