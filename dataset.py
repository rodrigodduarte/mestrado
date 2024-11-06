import torch
from torch.utils.data import DataLoader, random_split, Dataset, ConcatDataset
import pandas as pd
import numpy as np

import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import v2

import pytorch_lightning as pl

import PIL
import os

from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing


class CustomImageModule(pl.LightningDataModule):
    def __init__(self, train_dir, test_dir, shape, batch_size, num_workers):
        super().__init__()
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.shape = shape
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.image_transform = v2.Compose([
            v2.ToImage(),
            v2.Resize(self.shape, interpolation=PIL.Image.BILINEAR, antialias=False),
            v2.ToDtype(torch.uint8, scale=True),

            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(p=0.1),
            v2.RandomErasing(p=0.25),
            v2.RandAugment(num_ops=9, magnitude=5),

            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def setup(self, stage=None):
        """Setup training, validation, and test datasets."""
        if stage == "fit" or stage is None:
            # Load entire dataset with transforms applied
            entire_dataset = datasets.ImageFolder(root=self.train_dir, transform=self.image_transform)

            # Split dataset into training and validation (80-20 split)
            train_size = int(0.8 * len(entire_dataset))
            val_size = len(entire_dataset) - train_size
            self.train_ds, self.val_ds = random_split(entire_dataset, [train_size, val_size])

        if stage == "test" or stage is None:
            self.test_ds = datasets.ImageFolder(root=self.test_dir, transform=self.image_transform)

    def train_dataloader(self):
        """Return the training data loader."""
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )
    
    def val_dataloader(self):
        """Return the validation data loader."""
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )
    
    def test_dataloader(self):
        """Return the test data loader."""
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )
    

class CustomCSVModule(pl.LightningDataModule):
    def __init__(self, train_dir, test_dir, batch_size, num_workers):
        super().__init__()
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            entire_dataset = CSVFolder(root_dir=self.train_dir)

            # Split dataset into training and validation (80-20 split)
            train_size = int(0.8 * len(entire_dataset))
            val_size = len(entire_dataset) - train_size
            self.train_ds, self.val_ds = random_split(entire_dataset, [train_size, val_size])

        if stage == "test" or stage is None:
            self.test_ds = CSVFolder(root_dir=self.test_dir)

    def train_dataloader(self):
        """Return the training data loader."""
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )
    
    def val_dataloader(self):
        """Return the validation data loader."""
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )
    
    def test_dataloader(self):
        """Return the test data loader."""
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )

class CSVFolder(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Diretório com os arquivos CSV organizados por classe.
            transform (callable, optional): Transformações a serem aplicadas nas amostras.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.files = []
        self.labels = []
        
        # Percorre todas as subpastas e arquivos CSV
        for label in os.listdir(root_dir):
            label_dir = os.path.join(root_dir, label)
            if os.path.isdir(label_dir):
                for csv_file in os.listdir(label_dir):
                    if csv_file.endswith('.csv'):
                        self.files.append(os.path.join(label_dir, csv_file))
                        self.labels.append(label)
        
        # Codifica os rótulos em números (Label Encoding)
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.labels)
        self.labels = self.label_encoder.transform(self.labels)
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Carrega o arquivo CSV e converte em tensor
        csv_path = self.files[idx]
        features = pd.read_csv(csv_path, header=None).values.flatten()
        features = torch.tensor(features, dtype=torch.float32)
        
        label = self.labels[idx]
        
        if self.transform:
            features = self.transform(features)
        
        return features, label