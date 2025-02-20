import torch
import pytorch_lightning as pl
import numpy as np
from pytorch_lightning.callbacks import TQDMProgressBar, EarlyStopping, ModelCheckpoint

from model import CustomEnsembleModel
from dataset import CustomImageCSVModule

from callbacks import ImagesPerSecondCallback, EarlyStoppingAtSpecificEpoch

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
    hyperparams = load_hyperparameters('config_mlp.yaml')

    # Inicializar o wandb e acessar os parâmetros variáveis (do sweep)
    with wandb.init(project="swedish_mlp_ssn_384", config=config):
        config_sweep = wandb.config  # Acessar os parâmetros variáveis do sweep

        # Definir o data module com o diretório raiz e parâmetros do sweep
        data_module = CustomImageCSVModule(
            train_dir=hyperparams['TRAIN_DIR'],  # Diretório de treino
            test_dir=hyperparams['TEST_DIR'],
            shape=hyperparams['SHAPE'],        # Diretório de teste
            batch_size=config_sweep.batch_size,  # Batch size variável do sweep
            num_workers=hyperparams['NUM_WORKERS']  # Fixo
        )

        # Configurar o modelo com os parâmetros fixos e os variáveis do sweep
        model =CustomEnsembleModel(   
            epochs=hyperparams['MAX_EPOCHS'],               
            learning_rate=config_sweep.learning_rate,
            features_dim=hyperparams["FEATURES_DIM"],
            scale_factor = hyperparams['SCALE_FACTOR'],    
            drop_path_rate=hyperparams['DROP_PATH_RATE'],   
            num_classes=hyperparams['NUM_CLASSES'],         
            label_smoothing=hyperparams['LABEL_SMOOTHING'],
            optimizer_momentum=hyperparams['OPTIMIZER_MOMENTUM'],
            layer_scale = config_sweep.layer_scale
        )

        

        # Configurar o logger do W&B
        wandb_logger = WandbLogger(project="swedish_mlp_ssn_384")

        # Configurar o callback de Early Stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',  # Monitorar a acurácia de validação
            patience=15,              # Número de épocas para esperar antes de parar
            verbose=True,            # Exibir mensagens sobre o que está acontecendo
            mode='min'               # 'min' para minimizar a perda de validação
        )

        # Defina o limiar e a paciência
        threshold = 0.8  

        # Inicialize o callback
        early_stopping_threshold_callback = EarlyStoppingAtSpecificEpoch(stop_epoch=5, threshold=threshold)

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
                early_stopping_threshold_callback
            ]
        )

        # Treinando o modelo
        trainer.fit(model, data_module)

        # Testando o modelo
        trainer.test(model, data_module)

        # Finalizando
        wandb.finish()

if __name__ == "__main__":
    # Login no W&B
    wandb.login()

    # Configurar seeds para comportamento determinístico
    set_random_seeds()

    # Configurando o sweep
    sweep_config = {
        'method': 'random',  # método de busca aleatória
        'metric': {
            'name': 'val_accuracy',
            'goal': 'maximize'  # otimizar a acurácia de validação
        },
        'parameters': {
            'batch_size': {
                'values': [32]  # valores de batch size a serem testados
            },
            'learning_rate': {
                'min': 5e-5,           # valor mínimo da learning rate
                'max': 5e-4           # valor máximo da learning rate
            },
            'layer_scale':{
                'values': [0.7, 0.85, 1]
            }
        }
    }

    # Criar o sweep no W&B
    sweep_id = wandb.sweep(sweep_config, project="swedish_mlp_ssn_384")

    # Executar o sweep
    wandb.agent(sweep_id, function=train_model, count=200)

    wandb.finish()
