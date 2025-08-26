# data_utils.py

import torch
from torch.utils.data import DataLoader, Dataset, Subset
# Import the new dataset classes from torchvision
from torchvision.datasets import CIFAR100, STL10, MNIST, FashionMNIST, FGVCAircraft, Food101
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
import numpy as np
# --- THE DATASET REGISTRY ---
# This is the single source of truth for all dataset-specific information.
# To add a new dataset, just add a new entry here.
# --- THE DATASET REGISTRY ---
DATASET_REGISTRY = {
    "CIFAR100": {
        "dataset_class": CIFAR100,
        "num_classes": 100,
        "model_adapter": "small_3_channel",
        "input_size": 32,  # <-- This dataset has it
        "mean": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
    },
    "STL10": {
        "dataset_class": STL10,
        "num_classes": 10,
        "model_adapter": "small_3_channel",
        "input_size": 32,  # <-- This dataset has it
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2023, 0.1994, 0.2010),
    },
    "MNIST": {
        "dataset_class": MNIST,
        "num_classes": 10,
        "model_adapter": "small_1_channel",
        "input_size": 32,  # <-- THIS WAS MISSING. IT IS NOW ADDED.
        "mean": (0.1307,),
        "std": (0.3081,),
    },
    "FashionMNIST": {
        "dataset_class": FashionMNIST,
        "num_classes": 10,
        "model_adapter": "small_1_channel",
        "input_size": 32,  # <-- THIS WAS MISSING. IT IS NOW ADDED.
        "mean": (0.2860,),
        "std": (0.3530,),
    },
    "Aircraft": {
        "dataset_class": FGVCAircraft,
        "num_classes": 100,
        "model_adapter": "standard",
        "input_size": 224,  # <-- This dataset has it
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
    },
    "Food101": {
        "dataset_class": Food101,
        "num_classes": 101,
        "model_adapter": "standard",
        "input_size": 224,  # <-- This dataset has it
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
    }
}


def get_dataset_info(dataset_name):
    """Fetches the info dictionary for a given dataset from the registry."""
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset '{dataset_name}' not found in DATASET_REGISTRY.")
    return DATASET_REGISTRY[dataset_name]


def get_transforms(input_size, mean, std):
    """Creates train and test transforms for a given input size and normalization."""
    if input_size >= 224:
        train_trans = Compose([
            Resize((input_size, input_size)),
            ToTensor(),
            Normalize(mean, std)
        ])
    else:
        train_trans = Compose([
            # For smaller images, we resize first then crop
            Resize(input_size),
            ToTensor(),
            Normalize(mean, std)
        ])

    test_trans = Compose([
        Resize((input_size, input_size)),
        ToTensor(),
        Normalize(mean, std)
    ])

    return train_trans, test_trans


def get_dataloaders(dataset_name, train_transform, test_transform, train_classes, batch_size, eval_batch_size,
                    num_workers, pin_memory, nc_sample_size, test_classes=None):
    """
    This function now returns FOUR DataLoaders:
    1. train_loader (shuffled, for training)
    2. train_loader_eval (unshuffled, for calculating training set diagrams)
    3. train_loader_nc (unshuffled, from a fixed sample, for NC calculation)
    4. test_loader (unshuffled, for calculating validation loss/accuracy)
    """
    dataset_info = get_dataset_info(dataset_name)
    DatasetClass = dataset_info["dataset_class"]

    # --- CORRECTED LOGIC ---
    # STL10 is now correctly grouped with the datasets that use the 'split' argument.
    if dataset_name in ["Aircraft", "Food101", "Food101_Downsampled", "STL10"]:
        train_dataset = DatasetClass(root="./data", split="train", download=True, transform=train_transform)
        test_dataset = DatasetClass(root="./data", split="test", download=True, transform=test_transform)
    else: # For CIFAR, MNIST, etc. that use the 'train' boolean
        train_dataset = DatasetClass(root="./data", train=True, download=True, transform=train_transform)
        test_dataset = DatasetClass(root="./data", train=False, download=True, transform=test_transform)



    if dataset_name == 'CIFAR100':
        train_dataset = FilteredCIFAR100(train_dataset, train_classes)
        test_dataset = FilteredCIFAR100(test_dataset, train_classes)

    # 1. The standard, shuffled loader for training the model
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=pin_memory)

    # 2. An unshuffled loader for evaluating on the full training set (for diagrams)
    train_loader_eval = DataLoader(train_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers,
                                   pin_memory=pin_memory)

    # 3. A new loader from a fixed random sample of the training set (for NC metrics)
    num_train_samples = len(train_dataset)
    # Ensure sample size is not larger than the dataset
    sample_size = min(nc_sample_size, num_train_samples)

    # Use a fixed seed for reproducibility of the sample
    np.random.seed(42)
    fixed_indices = np.random.choice(num_train_samples, sample_size, replace=False)

    nc_train_subset = Subset(train_dataset, fixed_indices)
    train_loader_nc = DataLoader(nc_train_subset, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers,
                                 pin_memory=pin_memory)

    # 4. The standard loader for the test/validation set
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, train_loader_eval, train_loader_nc, test_loader

class FilteredCIFAR100(Dataset):
    def __init__(self, cifar100_dataset, keep_labels):
        self.cifar100_dataset = cifar100_dataset
        self.keep_labels = keep_labels
        self.label_map = {label: i for i, label in enumerate(keep_labels)}
        self.indices = [i for i, target in enumerate(self.cifar100_dataset.targets) if target in self.keep_labels]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        cifar_idx = self.indices[idx]
        image, label = self.cifar100_dataset[cifar_idx]
        return image, self.label_map.get(label, label)