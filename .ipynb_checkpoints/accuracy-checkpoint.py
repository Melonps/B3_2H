import torch
import torch.nn
from torch.utils import data
from torchvision import datasets, transforms

from vgg import vgg16


def accuracy(model, testloader):
    acc = 0
    for (input, target) in testloader:
        if torch.cuda.is_available():
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        with torch.no_grad():
            output = model(input)

        _value, prediction = output.topk(k=1, dim=1, largest=True, sorted=True)
        acc += (prediction.T == target).sum()

    return acc


if __name__ == "__main__":
    weight_path = "weight/checkpoint_299.pth"

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    testloader = data.DataLoader(
        dataset=datasets.CIFAR10(root="../data", train=False, transform=transform),
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    model = vgg16()
    model.cuda()
    try:
        state_dict = torch.load(weight_path)
    except FileNotFoundError:
        print("Weights not found.")
    model.load_state_dict(state_dict)
    model.eval()

    acc = accuracy(model, testloader)

    print("acc:{:.2f}%".format(acc / 100))

    print()
