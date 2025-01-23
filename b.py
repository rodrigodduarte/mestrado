import os
import torch
import pytorch_lightning as pl
import numpy as np

from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint

from model import CustomEnsembleModel, CustomModel
from dataset import CustomImageCSVModule, CustomImageModule
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

def empty_trash():
    trash_path = os.path.expanduser("~/.local/share/Trash")
    if os.path.exists(trash_path):
        subprocess.run(["rm", "-rf", f"{trash_path}/files/*", f"{trash_path}/info/*"])
        print("Lixeira esvaziada com sucesso no Linux.")

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
    hyperparams = load_hyperparameters('config_b.yaml')

    # Inicializar o W&B e acessar os parâmetros do sweep
    with wandb.init(project=hyperparams["PROJECT"], config=config):
        config_sweep = wandb.config

        # Configurar o DataModule
        data_module = CustomImageModule(
            train_dir=hyperparams['TRAIN_DIR'],
            test_dir=hyperparams['TEST_DIR'],
            shape=hyperparams['SHAPE'],
            batch_size=hyperparams['BATCH_SIZE'],
            num_workers=hyperparams['NUM_WORKERS']
        )

        # Configurar o modelo
        model = CustomModel(
            tmodel=hyperparams["TMODEL"],
            name_dataset= hyperparams["NAME_DATASET"],
            epochs=hyperparams['MAX_EPOCHS'],
            shape=hyperparams["SHAPE"],                              # Fixo
            learning_rate=float(config_sweep.learning_rate),       # Variável do sweep
            scale_factor=hyperparams['SCALE_FACTOR'],       # Fixo
            drop_path_rate=config_sweep.drop_path_rate,   # Fixo
            num_classes=hyperparams['NUM_CLASSES'],         # Fixo
            label_smoothing=config_sweep.label_smoothing,
            optimizer_momentum=(config_sweep.optimizer_momentum, 0.999)  # Fixo
        )

        # Configurar o logger do W&B
        wandb_logger = WandbLogger(project=hyperparams["PROJECT"])

        run_name = wandb.run.name
        checkpoint_path = f"{hyperparams['CHECKPOINT_PATH']}/{run_name}.ckpt"

        save_model_callback = SaveBestOrLastModelCallback(checkpoint_path)
        # checkpoint_callback = ModelCheckpoint(
        #     monitor="val_accuracy",
        #     dirpath=hyperparams["CHECKPOINT_PATH"],
        #     filename=f"{run_name}",
        #     save_top_k=1,
        #     mode="max",
        #     verbose=True
        # )

        # Callback de Early Stopping
        epoch_callback = EarlyStoppingAtSpecificEpoch(
            patience=2,
            threshold=1e-3,
            monitor="val_loss",
            mode="min",
            verbose=True
        )

        early_stop_callback = EarlyStopCallback(
            metric_name="val_loss",  # Métrica a ser monitorada
            threshold=0.5,          # Valor limite
            target_epoch=3          # Época em que verificar (índice começa em 0)
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
                       save_model_callback,
                       epoch_callback,
                       early_stop_callback]
        )

        # Treinamento
        trainer.fit(model, data_module)

        # Carregar o melhor modelo salvo após o treinamento
        model = CustomModel.load_from_checkpoint(checkpoint_path)

        # Testar o modelo carregado
        trainer.test(model, data_module)

        # Remover todos os arquivos na pasta de checkpoints
        checkpoint_dir = os.path.dirname(checkpoint_path)

        if os.path.exists(checkpoint_dir):
            for file_name in os.listdir(checkpoint_dir):
                file_path = os.path.join(checkpoint_dir, file_name)
                if os.path.isfile(file_path):  # Garante que é um arquivo
                    os.remove(file_path)
            print(f"Todos os arquivos foram removidos da pasta {checkpoint_dir}.")


        empty_trash()

        # Excluir a pasta do projeto
        project_dir = os.path.expanduser(hyperparams["PROJECT"])
        if os.path.exists(project_dir):
            shutil.rmtree(project_dir)
            print(f"A pasta {project_dir} foi excluída com sucesso.")
        else:
            print(f"A pasta {project_dir} não existe e não foi excluída.")  

        wandb.finish()


if __name__ == "__main__":
    # Login no W&B
    wandb.login()
    hyperparams = load_hyperparameters('config_b.yaml')

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

    # Executar o sweep
    wandb.agent(sweep_id, function=train_model, count=200)

    wandb.finish()
