import torch
import pytorch_lightning as pl
import numpy as np

from pytorch_lightning.callbacks import TQDMProgressBar, EarlyStopping, ModelCheckpoint
from pytorch_lightning.profilers import PyTorchProfiler

from model import CustomModel, CustomEnsembleModel
from dataset import CustomImageModule, CustomImageCSVModule
import config as config
from callbacks import  ImagesPerSecondCallback, StopOnPerfectTestAccuracyCallback

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

        # Definir o data module com o diretório raiz e parâmetros do sweep
        data_module = CustomImageCSVModule(
            train_dir=hyperparams['TRAIN_DIR'],  # Diretório de treino
            test_dir=hyperparams['TEST_DIR'],
            shape=hyperparams['SHAPE'],        # Diretório de teste
            batch_size=hyperparams["BATCH_SIZE"],  # Batch size variável do sweep
            num_workers=hyperparams['NUM_WORKERS']  # Fixo
        )

        # Configurar o modelo com os parâmetros fixos e os variáveis do sweep
        model =CustomEnsembleModel(
            name_dataset=hyperparams["NAME_DATASET"],
            shape=hyperparams["SHAPE"],
            epochs=hyperparams['MAX_EPOCHS'],               
            learning_rate=float(hyperparams["LEARNING_RATE"]),
            features_dim=hyperparams["FEATURES_DIM"],
            scale_factor = hyperparams['SCALE_FACTOR'],    
            drop_path_rate=hyperparams['DROP_PATH_RATE'],   
            num_classes=hyperparams['NUM_CLASSES'],         
            label_smoothing=hyperparams['LABEL_SMOOTHING'],
            optimizer_momentum=hyperparams['OPTIMIZER_MOMENTUM'],
            weight_decay=float(hyperparams["WEIGHT_DECAY"]),
            layer_scale = hyperparams["LAYER_SCALE"]
        )

        

        # Configurar o logger do W&B
        wandb_logger = WandbLogger(project=hyperparams["PROJECT"])

        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",               # Métrica a ser monitorada
            dirpath=hyperparams["CHECKPOINT_PATH"],             # Diretório onde o modelo será salvo
            filename="epoch{epoch}-val_loss{val_loss:.2f}",  # Nome do arquivo com época e val_loss
            save_top_k=1,                      # Salva apenas o melhor modelo
            mode="min",                        # Queremos minimizar o val_loss
            verbose=True                       # Ativa mensagens de log
        )

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
                ImagesPerSecondCallback(),
                # StopOnPerfectTestAccuracyCallback(),
                checkpoint_callback
            ]
        )

        # Treinando o modelo
        trainer.fit(model, data_module)

        # Obtenha o caminho do melhor modelo salvo
        best_model_path = checkpoint_callback.best_model_path
        print(f"Melhor modelo salvo em: {best_model_path}")

        # Carregar o modelo a partir do checkpoint
        best_model = CustomEnsembleModel.load_from_checkpoint(best_model_path)

        # Testando o modelo
        trainer.test(best_model, data_module)

        # Finalizando
        wandb.finish()

if __name__ == "__main__":
    # Login no WandB
    wandb.login()

    # Configurar seeds para comportamento determinístico
    set_random_seeds()

    # Treinar o modelo
    train_model()
