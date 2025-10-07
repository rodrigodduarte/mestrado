import os
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import pytorch_lightning as pl
from sklearn.model_selection import StratifiedKFold


# ================================================================
# Dataset: somente VETOR DE CARACTERÍSTICAS
# Aceita dois formatos de CSV:
#   (A) coluna 'features_path' -> caminho para .npy com shape (D,)
#   (B) colunas f_0, f_1, ..., f_{D-1}
# Em ambos os casos, é obrigatória a coluna 'label'.
# ================================================================
class FeaturesOnlyDataset(Dataset):
    def __init__(self, csv_path: str):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        if 'label' not in self.df.columns:
            raise ValueError("CSV deve conter coluna 'label'.")

        self.use_paths = 'features_path' in self.df.columns
        if not self.use_paths:
            self.feature_cols = [c for c in self.df.columns if c.startswith('f_')]
            if len(self.feature_cols) == 0:
                raise ValueError("CSV precisa ter 'features_path' OU colunas f_0, f_1, ...")
        else:
            self.feature_cols = None

        # Encoder de rótulos (caso venham como string)
        if self.df['label'].dtype == object:
            classes = sorted(self.df['label'].unique())
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
            self.df['label'] = self.df['label'].map(self.class_to_idx)
        else:
            self.class_to_idx = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        y = int(row['label'])
        if self.use_paths:
            x = np.load(row['features_path'])
        else:
            x = row[self.feature_cols].to_numpy(dtype=np.float32)
        x = np.asarray(x, dtype=np.float32)
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


# ================================================================
# DataModule estilo CustomImageCSVModule, mas para FEATURES
# Para treino/validação simples (split 80/20) e teste.
# ================================================================
class CustomFeaturesCSVModule(pl.LightningDataModule):
    def __init__(self,
                 train_csv: str,
                 test_csv: Optional[str],
                 batch_size: int,
                 num_workers: int,
                 val_split: float = 0.2,
                 seed: int = 42):
        super().__init__()
        self.train_csv = train_csv
        self.test_csv = test_csv
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.seed = seed

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            full_ds = FeaturesOnlyDataset(self.train_csv)
            if self.val_split <= 0.0:
                raise ValueError("val_split deve ser > 0.0 para criação do conjunto de validação.")
            n_full = len(full_ds)
            n_val = max(1, int(self.val_split * n_full))
            n_train = n_full - n_val
            self.train_ds, self.val_ds = random_split(
                full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(self.seed)
            )
        if stage == 'test' or stage is None:
            self.test_ds = FeaturesOnlyDataset(self.test_csv) if self.test_csv and os.path.exists(self.test_csv) else None

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        if self.test_ds is None:
            return None
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)


# ================================================================
# DataModule com K-Fold estratificado (compatível com pipelines KF)
# Mantém interface próxima ao CustomImageCSVModule_kf para facilitar plug-and-play.
# ================================================================
class CustomFeaturesCSVModule_kf(pl.LightningDataModule):
    def __init__(self,
                 train_csv: str,
                 test_csv: Optional[str],
                 batch_size: int,
                 num_workers: int,
                 n_splits: int = 5,
                 fold_idx: int = 0,
                 seed: int = 42):
        super().__init__()
        self.train_csv = train_csv
        self.test_csv = test_csv
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.n_splits = n_splits
        self.fold_idx = fold_idx
        self.seed = seed

        self._full_ds = None
        self._splits = None

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            self._full_ds = FeaturesOnlyDataset(self.train_csv)
            # Preparar rótulos para estratificação
            labels = []
            for i in range(len(self._full_ds.df)):
                labels.append(int(self._full_ds.df.iloc[i]['label']))
            labels = np.array(labels)

            skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
            indices = np.arange(len(self._full_ds))
            self._splits = list(skf.split(indices, labels))

            if self.fold_idx < 0 or self.fold_idx >= len(self._splits):
                raise ValueError(f"fold_idx {self.fold_idx} fora do intervalo [0, {len(self._splits)-1}].")

            train_idx, val_idx = self._splits[self.fold_idx]
            self.train_ds = Subset(self._full_ds, train_idx)
            self.val_ds = Subset(self._full_ds, val_idx)
            print(f"[Fold {self.fold_idx}] {len(train_idx)} treino / {len(val_idx)} validação.")

        if stage == 'test' or stage is None:
            self.test_ds = FeaturesOnlyDataset(self.test_csv) if self.test_csv and os.path.exists(self.test_csv) else None
            if self.test_ds is not None:
                print(f"[Test] {len(self.test_ds)} exemplos para teste.")

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        if self.test_ds is None:
            return None
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
