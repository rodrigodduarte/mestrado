import torch
from torch.utils.data import DataLoader, random_split, Dataset, ConcatDataset, Subset, WeightedRandomSampler
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


class CustomFeaturesFromFoldersModule_kf(pl.LightningDataModule):
    def __init__(self, train_dir, test_dir, shape, batch_size, num_workers,
                 n_splits=5, fold_idx=0, balance='none'):
        super().__init__()
        self.train_dir, self.test_dir = train_dir, test_dir
        self.batch_size, self.num_workers = batch_size, num_workers
        self.n_splits, self.fold_idx = int(n_splits), int(fold_idx)
        self.balance = balance
        self.kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        self.train_ds = self.val_ds = self.test_ds = None
        self.num_classes = None
        self.class_weights = None
        self._train_sample_weights = None

    def setup(self, stage=None):
        if stage in ('fit', None):
            full = FeaturesOnlyFromFoldersDataset(self.train_dir)
            self.num_classes = len(set(full.labels))
            idx = np.arange(len(full))
            splits = list(self.kf.split(idx))
            tr_idx, va_idx = splits[self.fold_idx]
            self.train_ds, self.val_ds = Subset(full, tr_idx), Subset(full, va_idx)

            # pesos por classe (opcional p/ desbalanceamento)
            train_labels = np.array([full.labels[i] for i in tr_idx], dtype=np.int64)
            counts = np.bincount(train_labels, minlength=self.num_classes).astype(np.float64)
            total = counts.sum(); eps = 1e-12
            cls_w = total / (self.num_classes * (counts + eps))
            self.class_weights = torch.tensor(cls_w, dtype=torch.float32)
            if self.balance in ('sampler', 'both'):
                self._train_sample_weights = torch.tensor(cls_w[train_labels], dtype=torch.double)

        if stage in ('test', None):
            self.test_ds = FeaturesOnlyFromFoldersDataset(self.test_dir)
            if self.num_classes is None:
                self.num_classes = len(set(self.test_ds.labels))

    def train_dataloader(self):
        if self.balance in ('sampler', 'both') and self._train_sample_weights is not None:
            sampler = WeightedRandomSampler(self._train_sample_weights,
                                            num_samples=len(self._train_sample_weights),
                                            replacement=True)
            return DataLoader(self.train_ds, batch_size=self.batch_size,
                              num_workers=self.num_workers, shuffle=False,
                              sampler=sampler, pin_memory=True)
        return DataLoader(self.train_ds, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False, pin_memory=True)

    def get_class_weights(self, device=None):
        if self.balance in ('weights', 'both') and self.class_weights is not None:
            return self.class_weights if device is None else self.class_weights.to(device)
        return None
