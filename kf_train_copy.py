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
from model import CustomEnsembleModel
from kf_data import CustomImageCSVModule_kf
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
    # Estrutura para armazenar métricas de todos os folds
    metrics_history = {}

    with wandb.init(project=hyperparams["PROJECT"], config=config):
        print(wandb.run.name)
        config_sweep = wandb.config

        wandb_logger = WandbLogger(project=hyperparams["PROJECT"])

        run_name = wandb.run.name
        run_dir = os.path.join("modelos_kf", run_name)
        os.makedirs(run_dir, exist_ok=True)

        for fold in range(k_splits):
            print(f"Processando Fold {fold+1}/{k_splits}")

            # Callback para salvar o melhor modelo deste fold
            fold_callback = ModelCheckpoint(
                dirpath=run_dir,
                filename=f"fold_{fold}_best_model",
                monitor="val_loss",
                mode="min",
                save_top_k=1
            )

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
                mlp_vector_model_scale=config_sweep.mlp_vector_model_scale)

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

            val_metrics = trainer.validate(model, data_module)[0]
            test_metrics = trainer.test(model, data_module)[0]

            # Armazenar métricas
            for metric_name, metric_value in val_metrics.items():
                if metric_name not in metrics_history:
                    metrics_history[metric_name] = []
                metrics_history[metric_name].append(metric_value)

            for metric_name, metric_value in test_metrics.items():
                if metric_name not in metrics_history:
                    metrics_history[metric_name] = []
                metrics_history[metric_name].append(metric_value)

            wandb.log({
                f"fold_{fold}/val_loss": val_metrics.get('val_loss'),
                f"fold_{fold}/val_accuracy": val_metrics.get('val_accuracy')
            })

        # Logar média e desvio padrão das métricas
        for metric_name, values in metrics_history.items():
            if isinstance(values[0], (int, float, np.float32, np.float64)):
                wandb.log({
                    f"{metric_name}_mean": np.mean(values),
                    f"{metric_name}_std": np.std(values)
                })

        wandb.finish()

if __name__ == "__main__":
    set_random_seeds()
    sweep_config = {
        'method': 'random',
        'metric': {'name': 'val_loss', 'goal': 'minimize'},
        'parameters': {
            'learning_rate': {'min': 0.00014997, 'max': 0.00014998, 'distribution': 'uniform'},
            'weight_decay': {'min': 4.4776e-7, 'max': 4.4777e-7, 'distribution': 'uniform'},
            'optimizer_momentum': {'min': 0.98135, 'max': 0.981366, 'distribution': 'uniform'},
            'mlp_vector_model_scale': {'min': 1.16582, 'max': 1.16583, 'distribution': 'uniform'},
            'layer_scale': {'min': 1.27079, 'max': 1.27080, 'distribution': 'uniform'},
            'drop_path_rate': {'min': 0.040087, 'max': 0.040088, 'distribution': 'uniform'},
            'label_smoothing': {'min': 0.08848, 'max': 0.08849, 'distribution': 'uniform'}
        }
    }
    sweep_id = wandb.sweep(sweep_config, project=load_hyperparameters('config.yaml')["PROJECT"])
    wandb.agent(sweep_id, function=train_model, count=200)
    wandb.finish()
