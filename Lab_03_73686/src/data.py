import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

BATCH_SIZE = 128

def get_loaders(data_dir="./data", seed=42):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])

    full_train = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
    test_set = datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)

    
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full_train, [55000, 5000], generator=generator)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_loaders()
    print("Train:", len(train_loader.dataset))
    print("Val:  ", len(val_loader.dataset))
    print("Test: ", len(test_loader.dataset))
    x, y = next(iter(train_loader))
    print("Batch shape:", x.shape, y.shape)