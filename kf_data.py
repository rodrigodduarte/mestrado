import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import v2
import PIL
import torch
import random
from sklearn.model_selection import KFold
from dataset import CustomImageWithFeaturesDataset


class CustomImageModule_kf(pl.LightningDataModule):
    def __init__(self, train_dir, test_dir, shape, batch_size, num_workers, n_splits=5, fold_idx=0):
        super().__init__()
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.shape = shape
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.n_splits = n_splits
        self.fold_idx = fold_idx  # Parâmetro para indicar o fold atual

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

        self.kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)  # Fixando a seed para garantir reprodutibilidade

    def setup(self, stage=None):
        """Configura os datasets de treino, validação e teste."""
        if stage == "fit" or stage is None:
            full_dataset = datasets.ImageFolder(root=self.train_dir, transform=self.image_transform)
            
            indices = list(range(len(full_dataset)))
            splits = list(self.kf.split(indices))
            if self.fold_idx >= len(splits):
                raise ValueError(f"Fold index {self.fold_idx} fora do intervalo permitido. Total de folds: {len(splits)}")
            train_indices, val_indices = splits[self.fold_idx]
            
            self.train_ds = torch.utils.data.Subset(full_dataset, train_indices)
            self.val_ds = torch.utils.data.Subset(full_dataset, val_indices)
            
            # Log dos índices para monitoramento
            print(f"[Fold {self.fold_idx + 1}] {len(train_indices)} exemplos para treino, {len(val_indices)} para validação.")

        if stage == "test" or stage is None:
            self.test_ds = datasets.ImageFolder(root=self.test_dir, transform=self.image_transform)

            print(f"[Test] {len(self.test_ds)} exemplos para teste.")

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)



class CustomImageCSVModule_kf(pl.LightningDataModule):
    def __init__(self, train_dir, test_dir, shape, batch_size, num_workers, n_splits=5, fold_idx=0):
        super().__init__()
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.shape = shape
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.n_splits = n_splits
        self.fold_idx = fold_idx  # Parâmetro para indicar o fold atual

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

        self.kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)  # Fixando a seed para garantir reprodutibilidade

    def setup(self, stage=None):
        """Configura os datasets de treino, validação e teste."""
        if stage == "fit" or stage is None:
            full_dataset = CustomImageWithFeaturesDataset(
                data_dir=self.train_dir,
                transform=self.image_transform
            )
            
            indices = list(range(len(full_dataset)))
            splits = list(self.kf.split(indices))
            if self.fold_idx >= len(splits):
                raise ValueError(f"Fold index {self.fold_idx} fora do intervalo permitido. Total de folds: {len(splits)}")
            train_indices, val_indices = splits[self.fold_idx]
            
            self.train_ds = torch.utils.data.Subset(full_dataset, train_indices)
            self.val_ds = torch.utils.data.Subset(full_dataset, val_indices)
            
            # Log dos índices para monitoramento
            print(f"[Fold {self.fold_idx + 1}] {len(train_indices)} exemplos para treino, {len(val_indices)} para validação.")

        if stage == "test" or stage is None:
            self.test_ds = CustomImageWithFeaturesDataset(
                data_dir=self.test_dir,
                transform=self.image_transform
            )
            print(f"[Test] {len(self.test_ds)} exemplos para teste.")

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
