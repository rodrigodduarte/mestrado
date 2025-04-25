import os
import shutil
import torch
import pytorch_lightning as pl
import numpy as np
import wandb
import yaml
import random
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from model import CustomModel
from kf_data import CustomImageModule_kf
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
def train_model(config=None):
    hyperparams = load_hyperparameters('config.yaml')
    k_splits = hyperparams['K_FOLDS']
    metrics_history = {}
    
    run_dir = os.path.join("modelos_ne_kf", f"{hyperparams['NAME_DATASET']}_{hyperparams['TMODEL']}")
    os.makedirs(run_dir, exist_ok=True)

    with wandb.init(project=hyperparams["PROJECT"], config=config):
        print(wandb.run.name)
        config_sweep = wandb.config

                
        wandb_logger = WandbLogger(project=hyperparams["PROJECT"])
        
        for fold in range(1):

            print(f"\n==================== Fold {fold+1}/{k_splits} ====================")

            fold_callback = ModelCheckpoint(
                dirpath=run_dir,
                filename=f"fold_{fold}_best_model",
                monitor="val_loss",
                mode="min",
                save_top_k=1
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

            print(f"\nTreinando Fold {fold+1}/{k_splits}")

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
                logger=wandb_logger,
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
        
    wandb.finish()

if __name__ == "__main__":
    set_random_seeds()
    sweep_config = {
        'method': 'random',
        'metric': {'name': 'val_loss', 'goal': 'minimize'},
        'parameters': {
            'learning_rate': {'min': 1e-5, 'max': 1e-4, 'distribution': 'log_uniform_values'},
            'weight_decay': {'min': 1e-6, 'max': 1e-3, 'distribution': 'log_uniform_values'},
            'optimizer_momentum': {'min': 0.92, 'max': 0.99, 'distribution': 'uniform'},
            'mlp_vector_model_scale': {'min': 0.8, 'max': 1.3, 'distribution': 'uniform'},
            'layer_scale': {'min': 0.75, 'max': 3, 'distribution': 'uniform'},
            'drop_path_rate': {'min': 0.0, 'max': 0.5, 'distribution': 'uniform'},
            'label_smoothing': {'min': 0.0, 'max': 0.2, 'distribution': 'uniform'}
        }
    }

    sweep_id = wandb.sweep(sweep_config, project=load_hyperparameters('config.yaml')["PROJECT"])
    wandb.agent(sweep_id, function=train_model, count=200)
    wandb.finish()
