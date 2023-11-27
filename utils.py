import torchvision
from torch.utils.data import DataLoader


def get_data(batch_size=32, num_workers=6):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32,padding=4),  # for cifar10
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_set = torchvision.datasets.CIFAR10("./data",download=True, train=True, transform=transforms)
    test_set = torchvision.datasets.CIFAR10("./data",download=True, train=False, transform=test_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader
