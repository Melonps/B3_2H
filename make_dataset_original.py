import os
import pickle
import warnings

import torch.nn.functional as F
from torch.utils import data
from torchvision import datasets, transforms
from tqdm import tqdm

warnings.simplefilter(action="ignore")

if __name__ == "__main__":
    save_data_folder = "../data/cifar10-modified/"

    if not os.path.exists(save_data_folder):
        os.makedirs(save_data_folder)

    transform = transforms.Compose([transforms.ToTensor()])
    trainloader = data.DataLoader(
        dataset=datasets.CIFAR10(root="../data", train=True, transform=transform, download=True),
        batch_size=5000,
        shuffle=False,
    )

    for i, (image, target) in enumerate(tqdm(trainloader)):
        # one_hot形式のtarget
        target = F.one_hot(target, num_classes=10).float()

        # np画像
        original_npimg = image.numpy()

        print(original_npimg.shape)
        start = 5000 * i
        end = 5000 * (i + 1) - 1

        with open(save_data_folder + 'original_target{}_{}'.format(start, end), 'wb') as f:
            pickle.dump(target, f)

        with open(save_data_folder + 'original_image{}_{}'.format(start, end), 'wb') as f:
            pickle.dump(original_npimg, f)
