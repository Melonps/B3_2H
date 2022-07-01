from PIL import Image
import os
import pickle
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


class MyDataset(Dataset):
    base_folder = "cifar10-modified"
    image_list = [
        "_image0_4999",
        "_image5000_9999",
        "_image10000_14999",
        "_image15000_19999",
        "_image20000_24999",
        "_image25000_29999",
        "_image30000_34999",
        "_image35000_39999",
        "_image40000_44999",
        "_image45000_49999",
    ]
    target_list = [
        "_target0_4999",
        "_target5000_9999",
        "_target10000_14999",
        "_target15000_19999",
        "_target20000_24999",
        "_target25000_29999",
        "_target30000_34999",
        "_target35000_39999",
        "_target40000_44999",
        "_target45000_49999",
    ]

    def __init__(self, root, use_ideal_target, percent, transform=None, target_tarnsform=None):
        super().__init__()

        self.transform = transform
        self.target_transform = target_tarnsform

        self.data = []
        self.targets = []

        for i, file_name in enumerate(self.image_list):
            if i < percent:
                file_name = "generated" + file_name
            else:
                file_name = "original" + file_name

            file_path = os.path.join(root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                image = pickle.load(f)
                self.data.append(image)
                print(len(self.data))

        for i, file_name in enumerate(self.target_list):
            if i < percent:
                if use_ideal_target:
                    file_name = "ideal" + file_name
                else:
                    file_name = "output" + file_name
            else:
                file_name = "original" + file_name

            file_path = os.path.join(root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                target = pickle.load(f)
                if 'output' in file_name:
                    
                    target = torch.cat(target).reshape(len(target), *target[0].shape)
                    print(target.shape)
                    target = F.softmax(target, dim=1)
                self.targets.append(target)

        print(len(self.data))
        self.data = np.vstack(self.data).transpose((0, 2, 3, 1))
        self.targets = torch.cat(self.targets).detach()

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray((img * 255).astype(np.uint8))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)
