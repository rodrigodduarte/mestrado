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
    

class CSVDataLoader(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Dataset para carregar arquivos CSV organizados por classe.
        
        Args:
            data_dir (str): Diretório contendo os dados das classes.
            transform (callable, optional): Transformações a serem aplicadas nos dados.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.data = []
        self.labels = []
        self.class_names = os.listdir(data_dir)
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.class_names)

        # Carregar dados de cada arquivo CSV
        for class_idx, class_name in enumerate(self.class_names):
            class_path = os.path.join(data_dir, class_name)
            for csv_file in os.listdir(class_path):
                if csv_file.endswith(".csv"):
                    csv_path = os.path.join(class_path, csv_file)
                    # Carregar os dados do CSV
                    df = pd.read_csv(csv_path)
                    features = df.values  # Assumindo que os dados estejam em um formato adequado
                    labels = [class_idx] * len(df)
                    self.data.append(features)
                    self.labels.append(labels)

        # Converter as listas para arrays
        self.data = torch.tensor(np.concatenate(self.data, axis=0), dtype=torch.float32)
        self.labels = torch.tensor(np.concatenate(self.labels, axis=0), dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)
        
        return sample, label


class CSVDataModule(pl.LightningDataModule):
    def __init__(self, train_dir, test_dir, batch_size=32, num_workers=4):
        """
        DataModule para carregar o dataset Swedish, com diretórios separados para treino e teste.
        
        Args:
            train_dir (str): Diretório com dados de treino.
            test_dir (str): Diretório com dados de teste.
            batch_size (int): Tamanho do batch.
            num_workers (int): Número de workers para o DataLoader.
        """
        super().__init__()
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Transformações (se necessário)
        self.transform = transforms.Compose([
            # Aqui, você pode adicionar transformações específicas para seus dados
        ])

    def setup(self, stage=None):
        """
        Configura os datasets de treino e teste.
        """
        self.train_dataset = CSVDataLoader(data_dir=self.train_dir, transform=self.transform)
        self.test_dataset = CSVDataLoader(data_dir=self.test_dir, transform=self.transform)

    def train_dataloader(self):
        """
        Retorna o DataLoader para o treino.
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        """
        Retorna o DataLoader para a validação/teste.
        """
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        """
        Retorna o DataLoader para o teste.
        """
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)