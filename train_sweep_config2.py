import torch
import pytorch_lightning as pl
import numpy as np

from pytorch_lightning.callbacks import TQDMProgressBar, EarlyStopping, ModelCheckpoint
from pytorch_lightning.profilers import PyTorchProfiler

from model import CustomModel
from dataset import CustomImageModule
import config as config
from callbacks import EarlyStoppingAtSpecificEpoch, ImagesPerSecondCallback, StopOnPerfectTestAccuracyCallback

import wandb
from pytorch_lightning.loggers import WandbLogger

import random
import yaml


# Carregar hiperparâmetros do arquivo YAML
def load_hyperparameters(file_path):
    with open(file_path, 'r') as file:
        hyperparams = yaml.safe_load(file)  # Carregar o YAML
    return hyperparams


def set_random_seeds():
    torch.backends.cudnn.deterministic = True
    random.seed(hash("setting random seeds") % 2**32 - 1)
    np.random.seed(hash("improves reproducibility") % 2**32 - 1)
    torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
    torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)


def train_model(config=None):
    hyperparams = load_hyperparameters('config2.yaml')

    # Inicializar o wandb e acessar os parâmetros variáveis (do sweep)
    with wandb.init(project=hyperparams["PROJECT"], config=config):
        config_sweep = wandb.config  # Acessar os parâmetros variáveis do sweep
        # Checar condição global antes de começar o treinamento

        # Definir o data module com os hiperparâmetros fixos e os do sweep
        data_module = CustomImageModule(
            train_dir=hyperparams['TRAIN_DIR'],  # Fixo
            test_dir=hyperparams['TEST_DIR'],    # Fixo
            shape=hyperparams['SHAPE'],          # Fixo
            batch_size=config_sweep.batch_size,  # Variável do sweep
            num_workers=hyperparams['NUM_WORKERS']  # Fixo
        )

        # Configurar o modelo com os parâmetros fixos e os variáveis do sweep
        model = CustomModel(
            tmodel=hyperparams["TMODEL"],
            name_dataset= hyperparams["NAME_DATASET"],
            shape = hyperparams["SHAPE"],
            epochs=hyperparams['MAX_EPOCHS'],               # Fixo
            learning_rate=config_sweep.learning_rate,       # Variável do sweep
            scale_factor=hyperparams['SCALE_FACTOR'],       # Fixo
            drop_path_rate=hyperparams['DROP_PATH_RATE'],   # Fixo
            num_classes=hyperparams['NUM_CLASSES'],         # Fixo
            label_smoothing=hyperparams['LABEL_SMOOTHING'],
            optimizer_momentum=hyperparams['OPTIMIZER_MOMENTUM']  # Fixo
        )  

        # Configurar o logger do W&B
        wandb_logger = WandbLogger(project=hyperparams["PROJECT"])

        # Configurar o callback de Early Stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',  # Monitorar a acurácia de validação
            patience=10,              # Número de épocas para esperar antes de parar
            verbose=True,            # Exibir mensagens sobre o que está acontecendo
            mode='min'               # 'max' para acurácia (procurando maximizar)
        )

        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",       # métrica a ser monitorada
            mode="min",               # "min" para menor val_loss
            filename="best-checkpoint-{epoch:02d}-{val_loss:.2f}",  # nome do arquivo do checkpoint
            save_top_k=1,             # salva apenas o melhor modelo
            dirpath="checkpoints/"    # diretório para salvar o checkpoint
        )

        # Defina o limiar e a paciência
        threshold = 0.8  
        stop_epoch = 8
        # Inicialize o callback
        early_stopping_threshold_callback_10 = EarlyStoppingAtSpecificEpoch(stop_epoch=10, threshold=0.8)
        early_stopping_threshold_callback_30 = EarlyStoppingAtSpecificEpoch(stop_epoch=35, threshold=0.67)
        # Configurar o Trainer do PyTorch Lightning
        trainer = pl.Trainer(
            logger=wandb_logger,    # W&B integration
            log_every_n_steps=10,
            accelerator=hyperparams['ACCELERATOR'],  # Fixo
            devices=hyperparams['DEVICES'],          # Fixo
            precision=hyperparams['PRECISION'],      # Fixo
            max_epochs=hyperparams['MAX_EPOCHS'],    # Fixo
            callbacks=[
                TQDMProgressBar(leave=True),
                early_stopping,
                ImagesPerSecondCallback(),
                early_stopping_threshold_callback_10,
                early_stopping_threshold_callback_30,
                checkpoint_callback,
                StopOnPerfectTestAccuracyCallback()
            ]
        )

        # Treinando o modelo
        trainer.fit(model, data_module)

        # Testando o modelo
        trainer.test(model, data_module)



        wandb.finish()

if __name__ == "__main__":
    # Login no W&B
    wandb.login()
    hyperparams = load_hyperparameters('config2.yaml')

    # Configurar seeds para comportamento determinístico
    set_random_seeds()

    # Configurando o sweep
    sweep_config = {
        'method': 'random',  # método de busca aleatória
        'metric': {
            'name': 'val_loss',
            'goal': 'minimize'  # otimizar a acurácia de validação
        },
        'parameters': {
            'batch_size': {
                'values': [32]  # valores de batch size a serem testados
            },
            'learning_rate': {
                'min': 9e-6,           # valor mínimo da learning rate
                'max': 7e-5            # valor máximo da learning rate
            }
        }
    }

    # Criar o sweep no W&B
    sweep_id = wandb.sweep(sweep_config, project=hyperparams["PROJECT"])

    # Executar o sweep
    wandb.agent(sweep_id, function=train_model, count=50)  # Executa o sweep com 10 variações

    wandb.finish()