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
from kf_data import CustomImageCSVModule_kf
from callbacks import (
    EarlyStoppingAtSpecificEpoch,
    SaveBestOrLastModelCallback,
    EarlyStopCallback
)

# Carregar hiperparâmetros do arquivo config2.yaml
def load_hyperparameters(file_path):
    with open(file_path, 'r') as file:
        hyperparams = yaml.safe_load(file)
    return hyperparams

# Configurar sementes para garantir reprodutibilidade
def set_random_seeds():
    torch.backends.cudnn.deterministic = True
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

# Função principal para treinamento com validação cruzada
def train_model():
    hyperparams = load_hyperparameters('config2.yaml')
    k_splits = hyperparams['K_FOLDS']
    best_val_loss = float('inf')
    best_checkpoint_path = None
    
    wandb.init(project=hyperparams["PROJECT"])

    for fold in range(k_splits):
        print(f"\nTreinando Fold {fold+1}/{k_splits}")

        data_module = CustomImageCSVModule_kf(
            train_dir=hyperparams['TRAIN_DIR'],
            test_dir=hyperparams['TEST_DIR'],
            shape=hyperparams['SHAPE'],
            batch_size=hyperparams['BATCH_SIZE'],
            num_workers=hyperparams['NUM_WORKERS'],
            n_splits=k_splits,
            fold_idx=fold
        )
        data_module.setup(stage='fit')

        # Determinar o checkpoint do fold anterior
        previous_checkpoint = f"{hyperparams['CHECKPOINT_PATH']}/fold_{fold}.ckpt" if fold > 0 else None
        
        if previous_checkpoint and os.path.exists(previous_checkpoint):
            print(f"Carregando modelo do fold anterior: {previous_checkpoint}")
            model = CustomEnsembleModel.load_from_checkpoint(previous_checkpoint)
        else:
            model = CustomEnsembleModel(
                tmodel=hyperparams["TMODEL"],
                name_dataset=hyperparams["NAME_DATASET"],
                shape=hyperparams["SHAPE"],
                epochs=hyperparams['MAX_EPOCHS'],
                learning_rate=float(hyperparams["LEARNING_RATE"]),
                features_dim=hyperparams["FEATURES_DIM"],
                scale_factor=hyperparams['SCALE_FACTOR'],
                drop_path_rate=hyperparams['DROP_PATH_RATE'],
                num_classes=hyperparams['NUM_CLASSES'],
                label_smoothing=hyperparams['LABEL_SMOOTHING'],
                optimizer_momentum=(hyperparams['OPTIMIZER_MOMENTUM'][0], hyperparams['OPTIMIZER_MOMENTUM'][1]),
                weight_decay=float(hyperparams['WEIGHT_DECAY']),
                layer_scale=hyperparams['LAYER_SCALE'],
                mlp_vector_model_scale=hyperparams['MLP_VECTOR_MODEL_SCALE']
            )

        checkpoint_path = f"{hyperparams['CHECKPOINT_PATH']}/fold_{fold+1}.ckpt"
        callbacks = [
            TQDMProgressBar(leave=True),
            SaveBestOrLastModelCallback(checkpoint_path),
            # EarlyStoppingAtSpecificEpoch(patience=4, threshold=1e-3, monitor="val_loss"),
            # EarlyStopCallback(metric_name="val_loss", threshold=0.5, target_epoch=3)
        ]

        wandb_logger = WandbLogger(project=hyperparams["PROJECT"], name=f"Fold_{fold+1}")

        trainer = pl.Trainer(
            logger=wandb_logger,
            log_every_n_steps=10,
            accelerator=hyperparams['ACCELERATOR'],
            devices=hyperparams['DEVICES'],
            precision=hyperparams['PRECISION'],
            max_epochs=hyperparams['MAX_EPOCHS'],
            callbacks=callbacks
        )

        trainer.fit(model, data_module)
        
        best_checkpoint_path = checkpoint_path
        print(f"Modelo salvo para o próximo fold: {best_checkpoint_path}")

    print(f"\nTreinamento finalizado. Melhor modelo salvo em: {best_checkpoint_path}")

    if best_checkpoint_path:
        print("\nIniciando teste final no melhor modelo...")
        best_model = CustomEnsembleModel.load_from_checkpoint(best_checkpoint_path)
        data_module.setup(stage='test')
        trainer.test(best_model, data_module)

    wandb.finish()

if __name__ == "__main__":
    set_random_seeds()
    train_model()
