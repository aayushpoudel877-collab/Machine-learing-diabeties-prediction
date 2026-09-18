from pathlib import Path
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

DEFAULT_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

def transforms_for(size: int, train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

def make_loader(root: str | Path, image_size: int, batch_size: int, train: bool, workers: int = 2):
    dataset = datasets.ImageFolder(root=str(root), transform=transforms_for(image_size, train))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=workers, pin_memory=torch.cuda.is_available())
    return loader, dataset
