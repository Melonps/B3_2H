import argparse
import math
import os
import pickle
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch import cuda, optim
from torch.utils import data
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
    weight_path = "weight/checkpoint_299.pth"
    save_data_folder = "../data/cifar10-modified/"

    if not os.path.exists(save_data_folder):
        os.makedirs(save_data_folder)

    transform = transforms.Compose([transforms.ToTensor()])
    trainloader = data.DataLoader(
        dataset=datasets.CIFAR10(root="../data", train=True, transform=transform),
        batch_size=1,
        shuffle=False,
    )

    lr = 0.05
    momentum = 0.9
    weight_decay = 5e-4
    eps = 10e-8

    model = GenerativeModel()

    try:
        state_dict = torch.load(weight_path)
    except FileNotFoundError:
        print("Weights not found.")

    model.classifier.load_state_dict(state_dict)
    model.eval()

    criterion = One_Hot_CrossEntropy()

    device = "cpu"
    if cuda.is_available():
        model.cuda()
        criterion.cuda()
        device = "cuda"

    # 重み更新の回数
    backprop_num = 100

    input = torch.tensor([[1.0]], requires_grad=True)
    input = input.to(device=device, non_blocking=True)

    ideal_targets = []
    output_targets = []
    generated_images = []

    for i, (image, target) in enumerate(tqdm(trainloader)):
        if i < args.start:
            continue
        if i >= args.end:
            break
        # 理想の出力の設定
        target = F.one_hot(target, num_classes=10)
        target = target * 0.9 + F.one_hot(torch.randint(0, 10, (1,)), num_classes=10) * 0.1
        ideal_targets.append(target)
        target = target.to(device=device, non_blocking=True)

        # 画像を重みとして設定
        image = image.to(device=device, non_blocking=True)
        image = torch.clamp(image, min=eps, max=1.0 - eps)
        image = torch.tan((image - 0.5) * math.pi)
        model.additional_layer.weight.data = image.view(-1, 1)

        optimizer = optim.SGD(model.additional_layer.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

        # 重み更新
        for j in range(backprop_num):
            output = model.generative(input)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 生成画像の取り出し
        generated_image = model.additional_layer.weight.data.view(1, 3, 32, 32).clone().cpu()
        generated_image = torch.arctan(generated_image)
        generated_image = generated_image / math.pi + 0.5
        generated_npimg = generated_image.numpy()
        generated_images.append(generated_npimg)

        # 生成画像をモデルに入力した際の出力の取り出し
        output = model.generative(input).clone().cpu()
        output_targets.append(output)

    ideal_targets = torch.cat(ideal_targets)
    output_targets = torch.cat(output_targets)
    generated_images = np.vstack(generated_images)

    with open(save_data_folder + 'ideal_target{}_{}'.format(args.start, args.end - 1), 'wb') as f:
        pickle.dump(ideal_targets, f)

    with open(save_data_folder + 'output_target{}_{}'.format(args.start, args.end - 1), 'wb') as f:
        pickle.dump(output_targets, f)

    with open(save_data_folder + 'generated_image{}_{}'.format(args.start, args.end - 1), 'wb') as f:
        pickle.dump(generated_images, f)
