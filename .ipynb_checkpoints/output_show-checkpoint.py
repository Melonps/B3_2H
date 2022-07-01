import torch
import torch.nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F

from vgg import vgg16


if __name__ == '__main__':
    fig_save_folder = 'output_fig/'
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    weight_path = 'weight_of_generative_training/checkpoint_{}_b{:02d}.pth'

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    test_set = datasets.CIFAR10(root='../data', train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=10,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    model = vgg16()
    model.cuda()

    for batch in [8, 16, 32]:
        for epoch in range(10):
            weight_path = 'weight_of_generative_training/checkpoint_{}_b{:02d}.pth'.format(epoch, batch)
            try:
                state_dict = torch.load(weight_path)
            except FileNotFoundError:
                print('Weights not found.')
            model.load_state_dict(state_dict)

            model.eval()

            for (images, _target) in test_loader:
                images = images.cuda(non_blocking=True)
                with torch.inference_mode():
                    outputs = model(images)

                outputs = F.softmax(outputs, dim=1).cpu().detach().numpy()

                for i, output in enumerate(outputs):
                    fig_save_path = fig_save_folder + 'label_e{}_b{:02d}_{}.png'.format(epoch, batch, i)
                    plt.bar(classes, output)
                    plt.title('epoch {}'.format(epoch+1))
                    plt.savefig(fig_save_path)
                    plt.close()

                break