import os
import torch
import pytorch_lightning as pl
import numpy as np

from torch.utils.data import Dataset, DataLoader

from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint

from sklearn.model_selection import StratifiedKFold

from model import CustomEnsembleModel
from dataset import CustomDataset_kf
from callbacks import EarlyStoppingAtSpecificEpoch, SaveBestOrLastModelCallback, EarlyStopCallback

import yaml
import wandb
from pytorch_lightning.loggers import WandbLogger
import random

import os
import platform
import subprocess
import ctypes

import shutil

# Carregar hiperparâmetros
def load_hyperparameters(path='config.yaml'):
    import yaml
    with open(path, 'r') as file:
        return yaml.safe_load(file)


def set_random_seeds():
    torch.backends.cudnn.deterministic = True
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def evaluate(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


# Função principal de treinamento com validação cruzada
def train_cv(config=None):
    set_random_seeds()
    hyperparams = load_hyperparameters('config.yaml')

    wandb.init(config=config)
    config = wandb.config

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = CustomDataset_kf(hyperparams['DATA_PATH'])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_accuracies = []  # Lista para armazenar a acurácia de cada fold

    for fold, (train_idx, val_idx) in enumerate(skf.split(dataset.filepaths, dataset.labels)):

        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=config.batch_size, shuffle=False)

        model = CustomEnsembleModel(
            num_classes=hyperparams['NUM_CLASSES'],
            drop_path_rate=config.drop_path_rate,
            learning_rate=config.learning_rate
        ).to(device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

        for epoch in range(config.epochs):
            train_epoch(model, train_loader, criterion, optimizer, device)
            val_accuracy = evaluate(model, val_loader, device)

            wandb.log({f"fold_{fold}_val_accuracy": val_accuracy, "epoch": epoch})

        fold_accuracies.append(val_accuracy)  # Armazena a acurácia final do fold

    # Calcula a média da acurácia dos folds
    mean_accuracy = np.mean(fold_accuracies)
    wandb.log({"mean_val_accuracy": mean_accuracy})  # Registra a média no WandB

    wandb.finish()


if __name__ == "__main__":
    set_random_seeds()  # Inicializa as sementes
    hyperparams = load_hyperparameters('config.yaml')

    sweep_config = {
        'method': 'random',
        'metric': {'name': 'val_accuracy', 'goal': 'maximize'},
        'parameters': {
            'learning_rate': {'values': [1e-4, 5e-5, 1e-5]},
            'batch_size': {'values': [16, 32]},
            'epochs': {'value': 50},
            'weight_decay': {'values': [1e-5, 1e-6]},
        }
    }

    sweep_id = wandb.sweep(sweep_config, project=hyperparams['PROJECT'])
    wandb.agent(sweep_id, train_cv, count=50)

    wandb.finish()
