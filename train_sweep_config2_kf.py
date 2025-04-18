import os
import torch
import pytorch_lightning as pl
import numpy as np
import wandb
import yaml
import random

from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from model import CustomEnsembleModel
from dataset import CustomImageCSVModule_kf
from callbacks import (
    EarlyStoppingAtSpecificEpoch,
    SaveBestOrLastModelCallback,
    EarlyStopCallback
)

# Carrega hiperparâmetros do arquivo config2.yaml
def load_hyperparameters(file_path):
    with open(file_path, 'r') as file:
        hyperparams = yaml.safe_load(file)
    return hyperparams

# Configuração das sementes para garantir reprodutibilidade
def set_random_seeds():
    torch.backends.cudnn.deterministic = True
    random.seed(hash("setting random seeds") % 2**32 - 1)
    np.random.seed(hash("improves reproducibility") % 2**32 - 1)
    torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
    torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)


# Função principal de treinamento com validação cruzada
def train_model_kf(config=None):
    hyperparams = load_hyperparameters('config2.yaml')
    k_splits = hyperparams.get('K_FOLDS', 5)
    print(k_splits)
    with wandb.init(project=hyperparams["PROJECT"], config=config):
        config_sweep = wandb.config

        for fold in range(k_splits):
            print(f"\nTreinando Fold {fold+1}/{k_splits}")

            # Inicializa o DataModule com os dados para o fold atual
            data_module = CustomImageCSVModule_kf(
                train_dir=hyperparams['TRAIN_DIR'],
                test_dir=hyperparams['TEST_DIR'],
                shape=hyperparams['SHAPE'],
                batch_size=hyperparams['BATCH_SIZE'],
                num_workers=hyperparams['NUM_WORKERS'],
                n_splits=k_splits
            )
            data_module.setup(stage='fit', fold_idx=fold)

            # Inicializa o modelo
            model = CustomEnsembleModel(
                tmodel=hyperparams["TMODEL"],
                name_dataset=hyperparams["NAME_DATASET"],
                shape=hyperparams["SHAPE"],
                epochs=hyperparams['MAX_EPOCHS'],
                learning_rate=float(config_sweep.learning_rate),
                features_dim=hyperparams["FEATURES_DIM"],
                scale_factor=hyperparams['SCALE_FACTOR'],
                drop_path_rate=config_sweep.drop_path_rate,
                num_classes=hyperparams['NUM_CLASSES'],
                label_smoothing=config_sweep.label_smoothing,
                optimizer_momentum=(config_sweep.optimizer_momentum, 0.999),
                weight_decay=float(config_sweep.weight_decay),
                layer_scale=config_sweep.layer_scale,
                mlp_vector_model_scale=config_sweep.mlp_vector_model_scale
            )

            # Caminho para salvar o checkpoint
            checkpoint_path = f"{hyperparams['CHECKPOINT_PATH']}/fold_{fold+1}.ckpt"

            # Definição dos callbacks
            callbacks = [
                TQDMProgressBar(leave=True),
                SaveBestOrLastModelCallback(checkpoint_path),
                EarlyStoppingAtSpecificEpoch(patience=4, threshold=1e-3, monitor="val_loss"),
                EarlyStopCallback(metric_name="val_loss", threshold=0.5, target_epoch=3)
            ]

            # Inicializa o logger do WandB
            wandb_logger = WandbLogger(project=hyperparams["PROJECT"], name=f"Fold_{fold+1}")

            # Configuração do treinador
            trainer = pl.Trainer(
                logger=wandb_logger,
                log_every_n_steps=10,
                accelerator=hyperparams['ACCELERATOR'],
                devices=hyperparams['DEVICES'],
                precision=hyperparams['PRECISION'],
                max_epochs=hyperparams['MAX_EPOCHS'],
                callbacks=callbacks
            )

            # Treinamento
            trainer.fit(model, data_module)

            # Testa o modelo salvo
            best_model = CustomEnsembleModel.load_from_checkpoint(checkpoint_path)
            trainer.test(best_model, data_module)

if __name__ == "__main__":
    set_random_seeds()  # Inicializa as sementes
    hyperparams = load_hyperparameters('config2.yaml')

    # Configuração do sweep
    sweep_config = {
        'method': 'random',
        'metric': {'name': 'val_loss', 'goal': 'minimize'},
        'parameters': {
            'learning_rate': {'min': 1e-5, 'max': 2e-4, 'distribution': 'uniform'},
            'weight_decay': {'min': 1e-7, 'max': 1e-6, 'distribution': 'uniform'},
            'optimizer_momentum': {'min': 0.92, 'max': 0.99, 'distribution': 'uniform'},
            'mlp_vector_model_scale': {'min': 0.8, 'max': 1.3, 'distribution': 'uniform'},
            'layer_scale': {'min': 0.5, 'max': 1.5, 'distribution': 'uniform'},
            'drop_path_rate': {'min': 0.0, 'max': 0.5, 'distribution': 'uniform'},
            'label_smoothing': {'min': 0.0, 'max': 0.2, 'distribution': 'uniform'}
        },
        'method': 'random',
        'metric': {'name': 'val_loss', 'goal': 'minimize'}
    }

    sweep_id = wandb.sweep(sweep_config, project=hyperparams["PROJECT"])
    wandb.agent(sweep_id, function=train_model_kf, count=200)
    wandb.finish()