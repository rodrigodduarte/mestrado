import os
import shutil
import torch
import pytorch_lightning as pl
import numpy as np
import yaml
import random
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from model import CustomEnsembleModel, CustomModel
from kf_data import CustomImageCSVModule_kf, CustomImageModule_kf
from callbacks import (
    EarlyStoppingAtSpecificEpoch,
    SaveBestOrLastModelCallback,
    EarlyStopCallback
)

# Carregar hiperparâmetros do arquivo config.yaml
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
    hyperparams = load_hyperparameters('config.yaml')
    k_splits = hyperparams['K_FOLDS']
    metrics_history = {}

    run_dir = os.path.join("modelos_kf", f"{hyperparams['NAME_DATASET']}_{hyperparams['TMODEL']}_ne")
    os.makedirs(run_dir, exist_ok=True)

    for fold in range(k_splits):
        print(f"\n==================== Fold {fold+1}/{k_splits} ====================")

        fold_callback = ModelCheckpoint(
            dirpath=run_dir,
            filename=f"fold_{fold}_best_model",
            monitor="val_loss",
            mode="min",
            save_top_k=1
        )

        model = CustomModel(
            tmodel=hyperparams["TMODEL"],
            name_dataset= hyperparams["NAME_DATASET"],
            epochs=hyperparams['MAX_EPOCHS'],
            shape=hyperparams["SHAPE"],                              # Fixo
            learning_rate=hyperparams['LEARNING_RATE'],      # Variável do sweep
            scale_factor=hyperparams['SCALE_FACTOR'],       # Fixo
            drop_path_rate=hyperparams['DROP_PATH_RATE'],   # Fixo
            num_classes=hyperparams['NUM_CLASSES'],         # Fixo
            label_smoothing=hyperparams['LABEL_SMOOTHING'],
            optimizer_momentum=(hyperparams['OPTIMIZER_MOMENTUM'], 0.999)  # Fixo
        )


        data_module = CustomImageModule_kf(
            train_dir=hyperparams['TRAIN_DIR'],
            test_dir=hyperparams['TEST_DIR'],
            shape=hyperparams['SHAPE'],
            batch_size=hyperparams['BATCH_SIZE'],
            num_workers=hyperparams['NUM_WORKERS'],        
            n_splits=k_splits,
            fold_idx=fold
        )
        data_module.setup(stage='fit')

        callbacks = [
            TQDMProgressBar(leave=True),
            fold_callback
        ]

        trainer = pl.Trainer(
            log_every_n_steps=10,
            accelerator=hyperparams['ACCELERATOR'],
            devices=hyperparams['DEVICES'],
            precision=hyperparams['PRECISION'],
            max_epochs=hyperparams['MAX_EPOCHS'],
            callbacks=callbacks
        )

        trainer.fit(model, data_module)

        best_model_path = fold_callback.best_model_path
        model = CustomModel.load_from_checkpoint(best_model_path)
        val_metrics = trainer.validate(model, data_module)[0]
        test_metrics = trainer.test(model, data_module)[0]

        for metric_name, metric_value in val_metrics.items():
            if metric_name not in metrics_history:
                metrics_history[metric_name] = []
            metrics_history[metric_name].append(metric_value)

        for metric_name, metric_value in test_metrics.items():
            if metric_name not in metrics_history:
                metrics_history[metric_name] = []
            metrics_history[metric_name].append(metric_value)

    print("\n==================== Métricas Finais ====================")
    for metric_name, values in metrics_history.items():
        if isinstance(values[0], (int, float, np.float32, np.float64)):
            mean = np.mean(values)
            std = np.std(values)
            print(f"{metric_name}: mean = {mean:.4f}, std = {std:.4f}")

if __name__ == "__main__":
    set_random_seeds()
    train_model()