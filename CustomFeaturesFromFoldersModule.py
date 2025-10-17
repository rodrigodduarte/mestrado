import torch
from torch.utils.data import DataLoader, random_split, Dataset, ConcatDataset, Subset
import pandas as pd
import numpy as np
from PIL import Image

import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import v2

import pytorch_lightning as pl

import PIL
import os

from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.model_selection import KFold


class FeaturesOnlyFromFoldersDataset(Dataset):
    def __init__(self, data_dir):
        """
        Carrega apenas os vetores de características armazenados em CSVs,
        organizados em subpastas por classe.

        Cada subdiretório deve representar uma classe e conter arquivos .csv,
        onde cada CSV representa um vetor de características (ex.: 1296 valores).

        Estrutura esperada:
        data_dir/
        ├── classe_1/
        │   ├── amostra_1.csv
        │   ├── amostra_2.csv
        │   └── ...
        ├── classe_2/
        │   ├── amostra_3.csv
        │   └── ...

        :param data_dir: Caminho da pasta raiz contendo as subpastas de classes
        """
        self.data_dir = data_dir
        self.csv_paths = []
        self.labels = []
        self.classes = sorted(os.listdir(data_dir))

        # Percorre todas as classes e coleta os CSVs
        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            for file_name in os.listdir(class_dir):
                if file_name.endswith('.csv'):
                    csv_path = os.path.join(class_dir, file_name)
                    self.csv_paths.append(csv_path)
                    self.labels.append(self.classes.index(class_name))

        if len(self.csv_paths) == 0:
            raise RuntimeError(f"Nenhum arquivo CSV encontrado em {data_dir}")

    def __len__(self):
        return len(self.csv_paths)

    def __getitem__(self, idx):
        """
        Retorna o vetor de características e o rótulo da classe.

        :param idx: Índice da amostra
        :return: (features, label)
        """
        csv_path = self.csv_paths[idx]
        label = self.labels[idx]

        try:
            # Lê o CSV como vetor 1D (sem cabeçalho)
            features = pd.read_csv(csv_path, header=None).values.flatten().astype(np.float32)
        except pd.errors.EmptyDataError:
            print(f"CSV vazio encontrado em {csv_path}. Substituindo por vetor de zeros.")
            features = np.zeros(1296, dtype=np.float32)  # dimensão padrão; ajuste conforme necessário

        features = torch.tensor(features, dtype=torch.float32)

        return features, label


class CustomFeaturesFromFoldersModule(pl.LightningDataModule):
    """
    DataModule para o conjunto de vetores de características organizados por classe em subpastas.

    Estrutura esperada:
    ├── train_dir/
    │   ├── classe_1/
    │   │   ├── amostra_1.csv
    │   │   ├── amostra_2.csv
    │   └── ...
    ├── test_dir/
    │   ├── classe_1/
    │   │   ├── amostra_X.csv
    │   └── ...
    """

    def __init__(self, train_dir, test_dir, batch_size, num_workers, val_split=0.2, seed=42):
        super().__init__()
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.seed = seed

    def setup(self, stage=None):
        """Configura os datasets de treino, validação e teste."""
        if stage == "fit" or stage is None:
            # Dataset completo de treino
            full_dataset = FeaturesOnlyFromFoldersDataset(data_dir=self.train_dir)

            # Divide em treino e validação
            val_size = int(self.val_split * len(full_dataset))
            train_size = len(full_dataset) - val_size

            self.train_ds, self.val_ds = random_split(
                full_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(self.seed)
            )

        if stage == "test" or stage is None:
            # Dataset de teste
            self.test_ds = FeaturesOnlyFromFoldersDataset(data_dir=self.test_dir)

    def train_dataloader(self):
        """DataLoader de treino."""
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True
        )

    def val_dataloader(self):
        """DataLoader de validação."""
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )

    def test_dataloader(self):
        """DataLoader de teste."""
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )
