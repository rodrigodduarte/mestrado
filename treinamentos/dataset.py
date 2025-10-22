import os

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, random_split, Dataset, Subset, WeightedRandomSampler

import pytorch_lightning as pl

import PIL
from PIL import Image

from torchvision import datasets
from torchvision.transforms import v2

from sklearn.model_selection import KFold


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

            # Geometric Transformations
            v2.RandomHorizontalFlip(p=0.1),
            v2.RandomVerticalFlip(p=0.1),
            v2.RandomRotation(degrees=30),  # Random rotation up to ±30°

            # Color Transformations
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            v2.RandomApply([v2.Grayscale(num_output_channels=3)], p=0.1),  # Grayscale with probability

            # Noise and Blur
            v2.RandomApply([v2.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))], p=0.2),  # Gaussian blur
            
            # Random Erasing
            v2.RandomErasing(p=0.25, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
            
            # Elastic Transformations
            v2.RandomPerspective(distortion_scale=0.2, p=0.2),

            # RandAugment
            v2.RandAugment(num_ops=9, magnitude=5),

            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # v2.ToImage(),
            # v2.Resize(self.shape, interpolation=PIL.Image.BILINEAR, antialias=False),
            # v2.ToDtype(torch.uint8, scale=True),

            # v2.RandomHorizontalFlip(p=0.1),
            # v2.RandomVerticalFlip(p=0.1),
            # v2.RandomErasing(p=0.25),
            # v2.RandAugment(num_ops=9, magnitude=5),
            # v2.ToDtype(torch.float32, scale=True),
            # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # # v2.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
            # # v2.Grayscale(num_output_channels=3)

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


class CustomImageWithFeaturesDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Construtor da classe que carrega imagens e vetores de características.

        :param data_dir: Diretório contendo subdiretórios das classes, com imagens e CSVs correspondentes
        :param transform: Transformações a serem aplicadas nas imagens
        """
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.csv_paths = []
        self.classes = sorted(os.listdir(data_dir))

        # Carregar caminhos das imagens e CSVs correspondentes
        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            for file_name in os.listdir(class_dir):
                if file_name.endswith('.png'):
                    img_path = os.path.join(class_dir, file_name)
                    csv_file = file_name.replace('.png', '.csv')
                    csv_path = os.path.join(class_dir, csv_file)
                    
                    if os.path.exists(csv_path):
                        self.image_paths.append(img_path)
                        self.csv_paths.append(csv_path)
                    else:
                        print(f"Arquivo CSV não encontrado para {file_name} em {class_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Carrega uma imagem e seu vetor de características correspondente.

        :param idx: Índice do item a ser recuperado
        :return: Imagem, vetor de características e rótulo da classe
        """
        # Carregar a imagem
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        # Carregar o vetor de características a partir do CSV
        csv_path = self.csv_paths[idx]
        
        try:
            # Lê o CSV como uma linha única de 1296 valores
            features = pd.read_csv(csv_path, header=None).values.flatten()
        except pd.errors.EmptyDataError:
            # Se o CSV estiver vazio, cria um vetor de zeros para evitar erros
            print(f"CSV vazio encontrado em {csv_path}. Substituindo por um vetor de zeros.")
            features = np.zeros(1296, dtype=np.float32)
        
        features = torch.tensor(features, dtype=torch.float32)

        # Determinar a classe da imagem
        label = self.classes.index(os.path.basename(os.path.dirname(image_path)))
        
        return image, features, label


class CustomImageCSVModule(pl.LightningDataModule):
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
        """Configura os datasets de treino, validação e teste."""
        if stage == "fit" or stage is None:
            # Carrega o dataset completo de treino e divide em treino/validação
            entire_dataset = CustomImageWithFeaturesDataset(
                data_dir=self.train_dir,
                transform=self.image_transform
            )
            
            train_size = int(0.8 * len(entire_dataset))
            val_size = len(entire_dataset) - train_size
            self.train_ds, self.val_ds = random_split(entire_dataset, [train_size, val_size])

        if stage == "test" or stage is None:
            # Carrega o dataset de teste
            self.test_ds = CustomImageWithFeaturesDataset(
                data_dir=self.test_dir,
                transform=self.image_transform
            )

    def train_dataloader(self):
        """Retorna o DataLoader de treino."""
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )
    
    def val_dataloader(self):
        """Retorna o DataLoader de validação."""
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )
    
    def test_dataloader(self):
        """Retorna o DataLoader de teste."""
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )


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
            print(f"[Fold {self.fold_idx}] {len(train_indices)} exemplos para treino, {len(val_indices)} para validação.")

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


class CustomImageModule_kf(pl.LightningDataModule):
    def __init__(self, train_dir, test_dir, shape, batch_size, num_workers, n_splits=5, fold_idx=0):
        super().__init__()
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.shape = shape
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.n_splits = n_splits
        self.fold_idx = fold_idx

        self.image_transform = v2.Compose([
            v2.ToImage(),
            v2.Resize(self.shape, interpolation=PIL.Image.BILINEAR, antialias=False),
            v2.ToDtype(torch.uint8, scale=True),

            # Geometric & Color transformations
            v2.RandomHorizontalFlip(p=0.1),
            v2.RandomVerticalFlip(p=0.1),
            v2.RandomRotation(degrees=30),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            v2.RandomApply([v2.Grayscale(num_output_channels=3)], p=0.1),
            v2.RandomApply([v2.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))], p=0.2),
            v2.RandomErasing(p=0.25, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
            v2.RandomPerspective(distortion_scale=0.2, p=0.2),
            v2.RandAugment(num_ops=9, magnitude=5),

            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            full_dataset = datasets.ImageFolder(root=self.train_dir, transform=self.image_transform)
            indices = list(range(len(full_dataset)))
            splits = list(self.kf.split(indices))

            if self.fold_idx >= len(splits):
                raise ValueError(f"Fold index {self.fold_idx} fora do intervalo permitido.")

            train_idx, val_idx = splits[self.fold_idx]
            self.train_ds = Subset(full_dataset, train_idx)
            self.val_ds = Subset(full_dataset, val_idx)

        if stage == "test" or stage is None:
            self.test_ds = datasets.ImageFolder(root=self.test_dir, transform=self.image_transform)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        

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
