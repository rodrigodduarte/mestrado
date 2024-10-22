import torch
import pytorch_lightning as pl
import numpy as np

from pytorch_lightning.callbacks import TQDMProgressBar, EarlyStopping
from pytorch_lightning.profilers import PyTorchProfiler

from model import CustomConvNeXtTiny, CustomConvNeXtLarge
from dataset import CustomImageModule
import config as config

import wandb
from pytorch_lightning.loggers import WandbLogger

import random

# Função de treinamento que será usada para o sweep
def train_model(config=None):
    # Inicia uma nova execução no wandb
    with wandb.init(config=config):
        config = wandb.config  # Atribui os parâmetros do sweep
        
        # Configura o DataModule com os hiperparâmetros do sweep
        data_module = CustomImageModule(
            train_dir=config.TRAIN_DIR,
            test_dir=config.TEST_DIR,
            shape=config.shape,
            batch_size=config.batch_size,
            num_workers=config.num_workers
        )

        # Instancia o modelo com os hiperparâmetros do sweep
        model = CustomConvNeXtTiny(
            epochs=config.epochs,
            learning_rate=config.learning_rate,
            scale_factor=config.scale_factor,
            drop_path_rate=config.drop_path_rate,
            num_classes=config.classes,
            label_smoothing=config.label_smoothing
        )
        
        # Configura o WandbLogger
        wandb_logger = WandbLogger(project="swedish classification with lightning")

        # Configura o Trainer do PyTorch Lightning
        trainer = pl.Trainer(
            logger=wandb_logger,     # Integração com W&B
            log_every_n_steps=50,    # Definir a frequência de logging
            profiler="simple",
            accelerator=config.ACCELERATOR,
            devices=config.DEVICES,
            precision=config.precision,
            max_epochs=config.epochs,
            callbacks=[TQDMProgressBar(leave=True), EarlyStopping(monitor="val_loss", patience=3)]
        )

        # Treina o modelo
        trainer.fit(model, data_module)

        # Testa o modelo
        trainer.test(model, data_module)


# Configuração do Sweep do Weights & Biases (grid search)
sweep_config = {
    'method': 'grid',  # Método de busca: 'grid' para grid search
    'metric': {
        'name': 'val_loss',  # Métrica a ser monitorada
        'goal': 'minimize'   # Minimizar a métrica
    },
    'parameters': {
        'learning_rate': {
            'values': [1e-5, 5e-5, 1e-4, 5e-4]  # Valores para o grid search
        },
        'batch_size': {
            'values': [8, 16]
        },
        'optimizer_momentum': {
            'values': [0.9, 0.999]
        },
        'drop_path_rate': {
            'values': [0.1]
        },
        'shape': {
            'values': [(224, 224)]  # Diferentes tamanhos de imagem
        },
        'epochs': {
            'values': [10, 20]  # Diferentes números de épocas
        },
        'classes': {
            'value': config.NUM_CLASSES  # Classe fixa do seu problema
        },
        'num_workers': {
            'value': config.NUM_WORKERS  # Número de workers fixo
        },
        'precision': {
            'value': config.PRECISION  # Precisão fixa
        },
        'scale_factor': {
            'value': config.SCALE_FACTOR  # Fator de escala fixo
        },
        'label_smoothing': {
            'value': config.LABEL_SMOOTHING  # Valor fixo de label smoothing
        },
        'TRAIN_DIR': {
            'value': config.TRAIN_DIR  # Diretório de treino fixo
        },
        'TEST_DIR': {
            'value': config.TEST_DIR  # Diretório de teste fixo
        },
        'ACCELERATOR': {
            'value': config.ACCELERATOR  # Acelerador fixo
        },
        'DEVICES': {
            'value': config.DEVICES  # Dispositivos fixos
        }
    }
}

# Função principal para iniciar o sweep
if __name__ == "__main__":
    # Fazer login no Wandb
    wandb.login()

    # Criar o sweep e definir a configuração
    sweep_id = wandb.sweep(sweep_config, project="swedish classification with lightning")
    
    # Iniciar o agente do sweep com a função de treinamento
    wandb.agent(sweep_id, train_model, count=10)  # 'count' é o número de execuções