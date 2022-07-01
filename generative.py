from generative_model import generative_model
import torch
from torch import optim
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
from loss import One_Hot_CrossEntropy
import torch.nn.functional as F
import math

if __name__ == '__main__':
    weight_path = 'weight/checkpoint_299_without_ddrop.pth'
    fig_save_folder = 'img/'

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    transform = transforms.Compose([transforms.ToTensor()])
    testset = datasets.CIFAR10(root='../data', train=False, transform=transform, download=True)
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

    try:
        state_dict = torch.load(weight_path)
    except FileNotFoundError:
        print('Weights not found.')

    model.classifier.load_state_dict(state_dict)
    model.eval()
    
    #わからないところ
    criterion = One_Hot_CrossEntropy()
    criterion = criterion.cuda()

    #生成する画像の枚数と重み更新の回数
    generate_num = 10
    backprop_num = 100

    input = torch.tensor([[1.0]], requires_grad=True)
    input = input.cuda(non_blocking=True)

    for i, (image, target) in enumerate(testloader):
        fig = plt.figure(figsize=(25, 6))
        if i >= generate_num:
            break

        #理想の出力の設定
        target = F.one_hot(target, num_classes=10)
        target = torch.tensor([[0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0]])
        target = target.cuda(non_blocking=True)
        
        #元画像の表示
        original_img = image[0].clone()
        original_npimg = original_img.numpy()
        ax1 = fig.add_subplot(1, 4, 1)
        plt.imshow(np.transpose(original_npimg, (1, 2, 0)))

        #画像を重みとして設定
        image = image.cuda(non_blocking=True)
        
        image = torch.clamp(image, min=eps, max=1.0-eps)
        image = torch.tan((image - 0.5)*math.pi)
        model.additional_layer.weight.data = image.view(-1, 1)

        losses = []
        optimizer = optim.SGD(model.additional_layer.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        
        #params : 更新したいパラメータを渡す.このパラメータは微分可能でないといけない.
        #lr : 学習率(learning rate). float型を渡す.
        #momentum : モーメンタム. float型を渡す.
        #dampening : モーメンタムの勢いを操作する. float型を渡す.
        #weight_decay : paramsのL2ノルムを正則化としてどれくらい加えるか. float型を渡す.
        #nesterov : nesterov momentumをモーメンタムとして適用するか.True or Falseを渡す.

        #重み更新
        for j in range(backprop_num):
            output = model.generative(input)
            loss = criterion(output, target)

            losses.append(loss.cpu().detach().numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #生成画像の表示
        generated_image = model.additional_layer.weight.data.view(3, 32, 32).clone().cpu()
        generated_image = torch.arctan(generated_image)
        generated_image = generated_image/math.pi + 0.5
        generated_npimg = generated_image.numpy()
        ax2 = fig.add_subplot(1, 4, 2)
        plt.imshow(np.transpose(generated_npimg, (1, 2, 0)))


        #生成画像をモデルに入力した際の出力のグラフの表示
        output = model.generative(input)
        output = F.softmax(output, dim=1).cpu().detach().numpy()[0]
        ax3 = fig.add_subplot(1, 4, 3)
        plt.bar(classes, output)

        #学習曲線の表示
        ax4 = fig.add_subplot(1, 4, 4)
        plt.plot(np.array([i for i in range(backprop_num)]), losses)
        plt.show()
        plt.close()
        
        #元画像の保存
        #fig_save_path = fig_save_folder + 'original_{}.png'.format(i)
        #plt.imshow(np.transpose(original_npimg, (1, 2, 0)))
        #plt.savefig(fig_save_path)
        #plt.close()

        #生成画像の保存
        #fig_save_path = fig_save_folder + 'generative_{}.png'.format(i)
        #plt.imshow(np.transpose(generated_npimg, (1, 2, 0)))
        #plt.savefig(fig_save_path)
        #plt.close()

        #出力の保存
        #fig_save_path = fig_save_folder + 'label_{}.png'.format(i)
        #plt.bar(classes, output)
        #plt.savefig(fig_save_path)
        #plt.close()