import torch
from torch.utils.data import DataLoader, random_split, Dataset, ConcatDataset
import pandas as pd

import torchvision
from torchvision import datasets
from torchvision.transforms import v2

import pytorch_lightning as pl

import PIL
import os


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
    

class CustomTabularDataset(Dataset):
    def __init__(self, csv_file):
        # Carrega o CSV e converte para tensor
        data = pd.read_csv(csv_file)
        self.features = torch.tensor(data.iloc[:, :-1].values, dtype=torch.float32)  # Todas as colunas, exceto a última (assumindo que a última é o rótulo)
        self.labels = torch.tensor(data.iloc[:, -1].values, dtype=torch.long)         # A última coluna como rótulo

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class CustomDataModule(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size, num_workers):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        """Setup training, validation, and test datasets."""
        if stage == "fit" or stage is None:
            train_dir = os.path.join(self.root_dir, 'train')
            train_datasets = [CustomTabularDataset(os.path.join(train_dir, csv_file))
                              for csv_file in os.listdir(train_dir) if csv_file.endswith('.csv')]
            entire_train_dataset = ConcatDataset(train_datasets)

            # Divide o dataset em treino e validação (80-20 split)
            train_size = int(0.8 * len(entire_train_dataset))
            val_size = len(entire_train_dataset) - train_size
            self.train_ds, self.val_ds = random_split(entire_train_dataset, [train_size, val_size])

        if stage == "test" or stage is None:
            test_dir = os.path.join(self.root_dir, 'test')
            test_datasets = [CustomTabularDataset(os.path.join(test_dir, csv_file))
                             for csv_file in os.listdir(test_dir) if csv_file.endswith('.csv')]
            self.test_ds = ConcatDataset(test_datasets)

    def train_dataloader(self):
        """Retorna o data loader para treino."""
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )
    
    def val_dataloader(self):
        """Retorna o data loader para validação."""
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )
    
    def test_dataloader(self):
        """Retorna o data loader para teste."""
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )