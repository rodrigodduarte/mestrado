import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision.transforms import v2
import PIL
import torch
import random
import numpy as np
from sklearn.model_selection import StratifiedKFold  # (alterado de KFold para estratificado)
from dataset import CustomImageWithFeaturesDataset

from torchvision import datasets
from sklearn.model_selection import KFold


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

        self._testimage_transform = v2.Compose([
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
            self.test_transform = v2.Compose([
                v2.ToImage(),
                v2.Resize(self.shape, interpolation=PIL.Image.BILINEAR, antialias=False),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            self.test_ds = datasets.ImageFolder(root=self.test_dir, transform=self.test_transform)
            self.num_classes = len(self.test_ds.classes) 
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




# =========================
# Helpers para rotulagem/pesos
# =========================
def _unwrap_dataset(ds):
    """Desempacota Subset(s) até o dataset base, retornando (base_ds, indices ou None)."""
    indices = None
    while isinstance(ds, Subset):
        indices = ds.indices if indices is None else [ds.indices[i] for i in indices]
        ds = ds.dataset
    return ds, indices

def _extract_labels(dataset):
    """
    Extrai rótulos do dataset (base ou Subset) de forma robusta.
    Tenta .targets / .labels / .y, depois .samples (ImageFolder), e por fim iterando (fallback).
    Retorna lista[int] com um rótulo por amostra na ordem do dataset recebido.
    """
    base, subset_idx = _unwrap_dataset(dataset)

    # 1) Atributos comuns
    for attr in ("targets", "labels", "y"):
        if hasattr(base, attr):
            base_labels = list(getattr(base, attr))
            if subset_idx is not None:
                return [int(base_labels[i]) for i in subset_idx]
            return [int(x) for x in base_labels]

    # 2) ImageFolder: .samples -> [(path, class_idx), ...]
    if hasattr(base, "samples"):
        base_labels = [c for _, c in base.samples]
        if subset_idx is not None:
            return [int(base_labels[i]) for i in subset_idx]
        return [int(x) for x in base_labels]

    # 3) Fallback: iterar (custo maior, mas compatível)
    if subset_idx is None:
        subset_idx = list(range(len(base)))
    out = []
    for i in subset_idx:
        item = base[i]
        # Suporta (img, label) ou (img, features, label)
        if isinstance(item, (list, tuple)):
            y = item[-1]
        else:
            raise RuntimeError("Não foi possível extrair rótulos do dataset.")
        out.append(int(y))
    return out

def _class_weights_from_counts(counts: np.ndarray, mode: str = "inv_freq", beta: float = 0.9999) -> np.ndarray:
    """
    Pesos por classe a partir das contagens.
      - inv_freq:  w_c = 1 / (count_c + eps)
      - effective_num (Cui et al.): w_c = (1 - beta) / (1 - beta^{count_c})
    Normaliza para média=1 (estabiliza a escala da loss).
    """
    eps = 1e-12
    counts = counts.astype(np.float64)
    if mode == "effective_num":
        counts = np.maximum(counts, 1.0)
        w = (1.0 - beta) / (1.0 - np.power(beta, counts))
    else:
        w = 1.0 / (counts + eps)
    w = w * (len(w) / (w.sum() + eps))
    return w


class CustomImageModule_kf_db(pl.LightningDataModule):
    def __init__(self, train_dir, test_dir, shape, batch_size, num_workers, n_splits=5, fold_idx=0):
        super().__init__()
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.shape = shape
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.n_splits = n_splits
        self.fold_idx = fold_idx  # Parâmetro para indicar o fold atual

        # ===== Flags opcionais (defaults seguros) =====
        self.use_weighted_sampler = False
        self.sampler_replacement = True
        self.use_class_weights = False
        self.weight_mode = "inv_freq"   # 'inv_freq' | 'effective_num'
        self.cb_beta = 0.9999

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

        self._testimage_transform = v2.Compose([
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

        # KFold -> StratifiedKFold (mantém proporções por classe)
        self.kf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        # Guardas
        self.class_weights = None
        self._train_labels = None  # rótulos do split de treino (para sampler)

    def setup(self, stage=None):
        """Configura os datasets de treino, validação e teste."""
        if stage == "fit" or stage is None:
            full_dataset = datasets.ImageFolder(root=self.train_dir, transform=self.image_transform)

            # Stratified split requer labels do conjunto completo
            full_labels = np.array(_extract_labels(full_dataset))
            indices = np.arange(len(full_labels))

            # Gera os splits estratificados
            splits = list(self.kf.split(indices, full_labels))
            if self.fold_idx >= len(splits):
                raise ValueError(f"Fold index {self.fold_idx} fora do intervalo permitido. Total de folds: {len(splits)}")
            train_indices, val_indices = splits[self.fold_idx]

            self.train_ds = Subset(full_dataset, train_indices)
            self.val_ds = Subset(full_dataset, val_indices)

            # Guarda rótulos do treino (para sampler e weights)
            self._train_labels = full_labels[train_indices]

            # Pesos por classe (opcional) — já calculamos aqui e deixamos disponível
            if self.use_class_weights:
                num_classes = int(self._train_labels.max()) + 1
                counts = np.bincount(self._train_labels, minlength=num_classes)
                w = _class_weights_from_counts(counts, mode=self.weight_mode, beta=self.cb_beta)
                self.class_weights = torch.tensor(w, dtype=torch.float32)
            else:
                self.class_weights = None

            print(f"[Fold {self.fold_idx + 1}] {len(train_indices)} exemplos para treino, {len(val_indices)} para validação.")

        if stage == "test" or stage is None:
            self.test_transform = v2.Compose([
                v2.ToImage(),
                v2.Resize(self.shape, interpolation=PIL.Image.BILINEAR, antialias=False),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            self.test_ds = datasets.ImageFolder(root=self.test_dir, transform=self.test_transform)
            self.num_classes = len(self.test_ds.classes)
            print(f"[Test] {len(self.test_ds)} exemplos para teste.")

    def train_dataloader(self):
        if self.use_weighted_sampler and self._train_labels is not None:
            counts = np.bincount(self._train_labels, minlength=int(self._train_labels.max()) + 1)
            inv_freq = 1.0 / np.maximum(counts, 1)
            sample_weights = inv_freq[self._train_labels]
            sampler = WeightedRandomSampler(
                weights=torch.as_tensor(sample_weights, dtype=torch.double),
                num_samples=len(sample_weights),
                replacement=self.sampler_replacement
            )
            return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers,
                              sampler=sampler, shuffle=False)
        # comportamento original
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


class CustomImageCSVModule_kf_db(pl.LightningDataModule):
    def __init__(self, train_dir, test_dir, shape, batch_size, num_workers, n_splits=5, fold_idx=0):
        super().__init__()
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.shape = shape
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.n_splits = n_splits
        self.fold_idx = fold_idx  # Parâmetro para indicar o fold atual

        # ===== Flags opcionais (defaults seguros) =====
        self.use_weighted_sampler = False
        self.sampler_replacement = True
        self.use_class_weights = False
        self.weight_mode = "inv_freq"   # 'inv_freq' | 'effective_num'
        self.cb_beta = 0.9999

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

        # KFold -> StratifiedKFold para manter proporções
        self.kf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        # Guardas
        self.class_weights = None
        self._train_labels = None

    def setup(self, stage=None):
        """Configura os datasets de treino, validação e teste."""
        if stage == "fit" or stage is None:
            full_dataset = CustomImageWithFeaturesDataset(
                data_dir=self.train_dir,
                transform=self.image_transform
            )

            # Labels completos para estratificar
            full_labels = np.array(_extract_labels(full_dataset))
            indices = np.arange(len(full_labels))

            splits = list(self.kf.split(indices, full_labels))
            if self.fold_idx >= len(splits):
                raise ValueError(f"Fold index {self.fold_idx} fora do intervalo permitido. Total de folds: {len(splits)}")
            train_indices, val_indices = splits[self.fold_idx]

            self.train_ds = Subset(full_dataset, train_indices)
            self.val_ds = Subset(full_dataset, val_indices)

            # Guarda rótulos de treino
            self._train_labels = full_labels[train_indices]

            # Pesos por classe (opcional)
            if self.use_class_weights:
                num_classes = int(self._train_labels.max()) + 1
                counts = np.bincount(self._train_labels, minlength=num_classes)
                w = _class_weights_from_counts(counts, mode=self.weight_mode, beta=self.cb_beta)
                self.class_weights = torch.tensor(w, dtype=torch.float32)
            else:
                self.class_weights = None

            print(f"[Fold {self.fold_idx + 1}] {len(train_indices)} exemplos para treino, {len(val_indices)} para validação.")

        if stage == "test" or stage is None:
            self.test_ds = CustomImageWithFeaturesDataset(
                data_dir=self.test_dir,
                transform=self.image_transform
            )
            print(f"[Test] {len(self.test_ds)} exemplos para teste.")

    def train_dataloader(self):
        if self.use_weighted_sampler and self._train_labels is not None:
            counts = np.bincount(self._train_labels, minlength=int(self._train_labels.max()) + 1)
            inv_freq = 1.0 / np.maximum(counts, 1)
            sample_weights = inv_freq[self._train_labels]
            sampler = WeightedRandomSampler(
                weights=torch.as_tensor(sample_weights, dtype=torch.double),
                num_samples=len(sample_weights),
                replacement=self.sampler_replacement
            )
            return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers,
                              sampler=sampler, shuffle=False)
        # comportamento original
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
