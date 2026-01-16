import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTImageProcessor


def build_transforms(dataset_name, train=False):
    proc = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    mean, std = proc.image_mean, proc.image_std
    resize_size, crop_size = 224, 224

    if train:
        tf = [
            transforms.Resize(int(resize_size * 1.15)),  
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip()
        ]
    else:
        tf = [
            transforms.Resize(resize_size),
            transforms.CenterCrop(crop_size)
        ]

    tf += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(tf)


def get_dataset(dataset_name, data_path, train=False):
    transform = build_transforms(dataset_name, train=train)
    dataset_name = dataset_name.lower()

    if dataset_name == "cifar10":
        dataset = datasets.CIFAR10(
            data_path,
            train=train,
            download=True,
            transform=transform
        )
    elif dataset_name == "imagenette":
        split = "train" if train else "val"
        dataset_dir = os.path.join(data_path, split)
        dataset = datasets.ImageFolder(dataset_dir, transform=transform)
    else:
        raise ValueError(
            f"Dataset '{dataset_name}' not supported. "
            f"Choose from ['cifar10', 'imagenette']"
        )

    return dataset


def get_dataloader(dataset, batch_size=128, shuffle=False, num_workers=4):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


def get_class_names(dataset_name):
    dataset_name = dataset_name.lower()

    if dataset_name == 'cifar10':
        return [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
    elif dataset_name == 'imagenette':
        return [
            'tench', 'English springer', 'cassette player', 'chain saw',
            'church', 'French horn', 'garbage truck', 'gas pump',
            'golf ball', 'parachute'
        ]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
