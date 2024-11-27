import torch
import pytorch_lightning as pl
import numpy as np

from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint

from model import CustomEnsembleModel
from dataset import CustomImageCSVModule
from callbacks import EarlyStoppingAtSpecificEpoch

import yaml
import wandb
from pytorch_lightning.loggers import WandbLogger
import random


# Carregar hiperparâmetros do arquivo YAML
def load_hyperparameters(file_path):
    with open(file_path, 'r') as file:
        hyperparams = yaml.safe_load(file)  # Carregar o YAML
    return hyperparams


# Configurar sementes para comportamento determinístico
def set_random_seeds():
    torch.backends.cudnn.deterministic = True
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)


# Função principal de treinamento
def train_model(config=None):
    hyperparams = load_hyperparameters('config.yaml')

    # Inicializar o W&B e acessar os parâmetros do sweep
    with wandb.init(project=hyperparams["PROJECT"], config=config):
        config_sweep = wandb.config

        # Configurar o DataModule
        data_module = CustomImageCSVModule(
            train_dir=hyperparams['TRAIN_DIR'],
            test_dir=hyperparams['TEST_DIR'],
            shape=hyperparams['SHAPE'],
            batch_size=hyperparams['BATCH_SIZE'],
            num_workers=hyperparams['NUM_WORKERS']
        )

        # Configurar o modelo
        model = CustomEnsembleModel(
            name_dataset=hyperparams["NAME_DATASET"],
            shape=hyperparams["SHAPE"],
            epochs=hyperparams['MAX_EPOCHS'],
            learning_rate=float(config_sweep.learning_rate),
            features_dim=hyperparams["FEATURES_DIM"],
            scale_factor=hyperparams['SCALE_FACTOR'],
            drop_path_rate=config_sweep.drop_path_rate,
            num_classes=hyperparams['NUM_CLASSES'],
            label_smoothing=config_sweep.label_smoothing,
            optimizer_momentum=(config_sweep.optimizer_momentum, 0.999),  # AdamW usa dois betas
            weight_decay=float(config_sweep.weight_decay),
            layer_scale=config_sweep.layer_scale,
            mlp_vector_model_scale=config_sweep.mlp_vector_model_scale
        )

        # Configurar o logger do W&B
        wandb_logger = WandbLogger(project=hyperparams["PROJECT"])

        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath=hyperparams["CHECKPOINT_PATH"],
            filename="epoch{epoch}-val_loss{val_loss:.2f}",
            save_top_k=1,
            mode="min",
            verbose=True
        )

        # Callback de Early Stopping
        early_stopping_callback = EarlyStoppingAtSpecificEpoch(
            patience=5,
            threshold=1e-3,
            monitor="val_loss",
            mode="min",
            verbose=True
        )

        # Configurar o Trainer
        trainer = pl.Trainer(
            logger=wandb_logger,
            log_every_n_steps=10,
            accelerator=hyperparams['ACCELERATOR'],
            devices=hyperparams['DEVICES'],
            precision=hyperparams['PRECISION'],
            max_epochs=hyperparams['MAX_EPOCHS'],
            callbacks=[TQDMProgressBar(leave=True), 
                       checkpoint_callback,
                       early_stopping_callback]
        )

        # Treinamento
        trainer.fit(model, data_module)

        # Testar o modelo
        trainer.test(model, data_module)

        wandb.finish()


if __name__ == "__main__":
    # Login no W&B
    wandb.login()
    hyperparams = load_hyperparameters('config.yaml')

    # Configurar sementes
    set_random_seeds()

    # Configurar o sweep
    sweep_config = {
        'method': 'random',  # Random search
        'metric': {
            'name': 'val_loss',
            'goal': 'minimize'
        },
        'parameters': {
            'learning_rate': {
                'min': 1e-5,
                'max': 1e-4,
                'distribution': 'uniform'
            },
            'weight_decay': {
                'min': 1e-9,
                'max': 1e-6,
                'distribution': 'uniform'
            },
            'optimizer_momentum': {
                'min': 0.85,
                'max': 0.99,
                'distribution': 'uniform'
            },
            'mlp_vector_model_scale': {
                'min': 0.5,
                'max': 1.5,
                'distribution': 'uniform'
            },
            'layer_scale': {
                'min': 0.5,
                'max': 1.5,
                'distribution': 'uniform'
            },
            'drop_path_rate': {
                'min': 0.0,
                'max': 0.5,
                'distribution': 'uniform'
            },
            'label_smoothing': {
                'min': 0.0,
                'max': 0.2,
                'distribution': 'uniform'
            }
        }
    }

    # Criar o sweep
    sweep_id = wandb.sweep(sweep_config, project=hyperparams["PROJECT"])

    # Executar o sweep
    wandb.agent(sweep_id, function=train_model, count=200)

    wandb.finish()
