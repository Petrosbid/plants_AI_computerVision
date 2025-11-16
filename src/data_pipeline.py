# src/data_pipeline.py

import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from sklearn.model_selection import train_test_split
import os
import random


def get_data_augmentation_transforms(img_height, img_width):
    """
    Returns a PyTorch transform for data augmentation.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop((img_height, img_width)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])


def get_validation_transforms(img_height, img_width):
    """
    Returns a PyTorch transform for validation (no augmentation).
    """
    return transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])


def build_data_pipeline(config):
    """
    Builds training and validation data pipelines using PyTorch DataLoader.

    Args:
        config (dict): Dictionary loaded from params.yaml

    Returns:
        tuple: (train_loader, val_loader, class_names)
    """
    data_dir = config['data_dir']
    img_height = config['input_shape'][0]
    img_width = config['input_shape'][1]
    batch_size = config['batch_size']
    val_split = config['validation_split']
    seed = config['seed']

    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"Loading data from: {data_dir}")

    # Get all data with validation transforms initially
    full_dataset = datasets.ImageFolder(root=data_dir, transform=get_validation_transforms(img_height, img_width))

    # Get class names
    class_names = full_dataset.classes
    print(f"Found {len(class_names)} classes: {class_names}")

    # Split dataset into train and validation
    indices = list(range(len(full_dataset)))
    train_indices, val_indices = train_test_split(
        indices, test_size=val_split, random_state=seed, stratify=full_dataset.targets
    )

    # Create datasets with appropriate transforms
    train_dataset = datasets.ImageFolder(
        root=data_dir,
        transform=get_data_augmentation_transforms(img_height, img_width)
    )
    val_dataset = datasets.ImageFolder(
        root=data_dir,
        transform=get_validation_transforms(img_height, img_width)
    )

    # Subset the datasets to their respective indices
    train_dataset = Subset(train_dataset, train_indices)
    val_dataset = Subset(val_dataset, val_indices)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,  # Adjust based on your system
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,  # Adjust based on your system
        pin_memory=True if torch.cuda.is_available() else False
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    return train_loader, val_loader, class_names