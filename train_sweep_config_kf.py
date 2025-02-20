import os
import torch
import pytorch_lightning as pl
import numpy as np

from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from model import CustomEnsembleModel
from dataset import CustomImageCSVModule_kf  # Versão com validação cruzada
from callbacks import EarlyStoppingAtSpecificEpoch, SaveBestOrLastModelCallback, EarlyStopCallback

import yaml
import wandb
import random
import shutil

# Carregar hiperparâmetros do arquivo YAML
def load_hyperparameters(file_path):
    with open(file_path, 'r') as file:
        hyperparams = yaml.safe_load(file)  # Carregar o YAML
    return hyperparams

# Configurar sementes para reprodutibilidade
def set_random_seeds():
    torch.backends.cudnn.deterministic = True
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

# Função principal de treinamento com validação cruzada
def train_model_kf(config=None):
    hyperparams = load_hyperparameters('config.yaml')
    k_splits = hyperparams.get('K_FOLDS', 5)  # Número de folds para validação cruzada

    # Inicializar o W&B e acessar os parâmetros do sweep
    with wandb.init(project=hyperparams["PROJECT"], config=config):
        config_sweep = wandb.config  # Parâmetros do sweep

        for fold in range(k_splits):
            print(f"\n########### Treinando Fold {fold+1}/{k_splits} ###########\n")

            # Configurar o DataModule para o fold atual
            data_module = CustomImageCSVModule_kf(
                train_dir=hyperparams['TRAIN_DIR'],
                test_dir=hyperparams['TEST_DIR'],
                shape=hyperparams['SHAPE'],
                batch_size=hyperparams['BATCH_SIZE'],
                num_workers=hyperparams['NUM_WORKERS'],
                n_splits=k_splits
            )
            data_module.setup(fold_idx=fold)

            # Configurar o modelo
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

            # Configurar o logger do W&B para cada fold
            wandb_logger = WandbLogger(project=hyperparams["PROJECT"], name=f"Fold_{fold+1}")

            # Definir caminho do modelo salvo para cada fold
            checkpoint_path = f"{hyperparams['CHECKPOINT_PATH']}/fold_{fold+1}.ckpt"

            save_model_callback = SaveBestOrLastModelCallback(checkpoint_path)

            # Callback de Early Stopping
            epoch_callback = EarlyStoppingAtSpecificEpoch(
                patience=4,
                threshold=1e-3,
                monitor="val_loss",
                mode="min",
                verbose=True
            )

            early_stop_callback = EarlyStopCallback(
                metric_name="val_loss",  # Métrica a ser monitorada
                threshold=0.5,
                target_epoch=3
            )

            # Configurar o Trainer
            trainer = pl.Trainer(
                logger=wandb_logger,
                log_every_n_steps=10,
                accelerator=hyperparams['ACCELERATOR'],
                devices=hyperparams['DEVICES'],
                precision=hyperparams['PRECISION'],
                max_epochs=hyperparams['MAX_EPOCHS'],
                callbacks=[TQDMProgressBar(leave=True), save_model_callback, epoch_callback, early_stop_callback]
            )

            # Treinamento
            trainer.fit(model, data_module)

            # Carregar o melhor modelo salvo após o treinamento
            best_model = CustomEnsembleModel.load_from_checkpoint(checkpoint_path)

            # Testar o modelo carregado
            trainer.test(best_model, data_module)

            print(f"Fold {fold+1} concluído. Melhor modelo salvo em {checkpoint_path}")

        wandb.finish()


if __name__ == "__main__":
    # Login no W&B
    wandb.login()
    hyperparams = load_hyperparameters('config.yaml')

    # Configurar sementes
    set_random_seeds()

    # Configurar o sweep
    sweep_config = {
        'method': 'random',  # Busca aleatória
        'metric': {
            'name': 'val_loss',
            'goal': 'minimize'
        },
        'parameters': {
            'learning_rate': {
                'min': 1e-5,
                'max': 2e-4,
                'distribution': 'uniform'
            },
            'weight_decay': {
                'min': 1e-7,
                'max': 1e-6,
                'distribution': 'uniform'
            },
            'optimizer_momentum': {
                'min': 0.92,
                'max': 0.99,
                'distribution': 'uniform'
            },
            'mlp_vector_model_scale': {
                'min': 0.8,
                'max': 1.3,
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

    # Executar o sweep com validação cruzada
    wandb.agent(sweep_id, function=train_model_kf, count=200)

    wandb.finish()
