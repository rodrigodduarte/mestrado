#!/usr/bin/env python3
"""
train_vector_sweep_config.py
─────────────────────────────
Sweep no Weights & Biases para **modelo baseado apenas no vetor de características**
(MLP), padronizado a partir do `train_sweep_config.py` fornecido.

• Usa `CustomImageCSVModule` (não K-Fold), igual ao script-base.
• Substitui `CustomEnsembleModel` por `CustomVectorModel`.
• Mantém callbacks, logger, e política de salvar/carregar melhor checkpoint pelo nome do run.
• Após o teste, limpa a pasta de checkpoints e esvazia a lixeira (Linux), como no script-base.

Execução:
    conda activate mestrado
    python train_vector_sweep_config.py

Pré-requisitos:
  - config.yaml com chaves: PROJECT, TRAIN_DIR, TEST_DIR, SHAPE, BATCH_SIZE,
    NUM_WORKERS, ACCELERATOR, DEVICES, PRECISION, MAX_EPOCHS, NAME_DATASET,
    FEATURES_DIM, NUM_CLASSES, SCALE_FACTOR, CHECKPOINT_PATH
  - model.CustomVectorModel
  - dataset.CustomImageCSVModule
  - callbacks.{EarlyStoppingAtSpecificEpoch, SaveBestOrLastModelCallback, EarlyStopCallback}
"""

import os
import platform
import subprocess
import shutil
import random
import yaml
import numpy as np
import torch
import pytorch_lightning as pl

from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger

import wandb

# Projeto
from model import CustomVectorModel
from dataset import CustomImageCSVModule
from callbacks import (
    EarlyStoppingAtSpecificEpoch,
    SaveBestOrLastModelCallback,
    EarlyStopCallback,
)


def empty_trash():
    """Esvazia a lixeira no Linux (mesmo comportamento do script-base)."""
    if platform.system() == "Linux":
        trash_path = os.path.expanduser("~/.local/share/Trash")
        if os.path.exists(trash_path):
            subprocess.run([
                "rm",
                "-rf",
                f"{trash_path}/files/*",
                f"{trash_path}/info/*",
            ])
            print("Lixeira esvaziada com sucesso no Linux.")


def load_hyperparameters(file_path: str = "config.yaml"):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def set_random_seeds(seed: int = 42):
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_model(config=None):
    hp = load_hyperparameters("config.yaml")

    # Inicializa W&B e captura parâmetros do sweep
    with wandb.init(project=hp["PROJECT"], config=config):
        cfg = wandb.config

        # DataModule (igual ao script-base: sem K-Fold)
        dm = CustomImageCSVModule(
            train_dir=hp["TRAIN_DIR"],
            test_dir=hp["TEST_DIR"],
            shape=hp["SHAPE"],
            batch_size=hp["BATCH_SIZE"],
            num_workers=hp["NUM_WORKERS"],
            # Caso seu datamodule permita, você pode sinalizar para usar somente vetores:
            # use_vectors_only=True,
        )

        # Modelo: MLP apenas para vetor de características
        model = CustomVectorModel(
            tmodel=hp.get("TMODEL", "vector"),  # compatibilidade com assinatura
            name_dataset=hp["NAME_DATASET"],
            shape=hp["SHAPE"],  # pode ser ignorado internamente pelo MLP
            epochs=hp["MAX_EPOCHS"],
            learning_rate=float(cfg.learning_rate),
            features_dim=hp["FEATURES_DIM"],
            scale_factor=hp.get("SCALE_FACTOR", 1.0),
            drop_path_rate=float(cfg.drop_path_rate),  # mantido por padronização (pode ser ignorado)
            num_classes=hp["NUM_CLASSES"],
            label_smoothing=float(cfg.label_smoothing),
            optimizer_momentum=(float(cfg.optimizer_momentum), 0.999),  # AdamW betas
            weight_decay=float(cfg.weight_decay),
            layer_scale=float(cfg.layer_scale),
            mlp_vector_model_scale=float(cfg.mlp_vector_model_scale),
        )

        # Logger
        wandb_logger = WandbLogger(project=hp["PROJECT"])

        # Salvar melhor modelo com nome do run (mesmo padrão do base)
        run_name = wandb.run.name
        ckpt_path = f"{hp['CHECKPOINT_PATH']}/{run_name}.ckpt"
        save_model_cb = SaveBestOrLastModelCallback(ckpt_path)

        # Callbacks de parada/monitoramento (iguais ao base)
        epoch_cb = EarlyStoppingAtSpecificEpoch(
            patience=2,
            threshold=1e-3,
            monitor="val_loss",
            mode="min",
            verbose=True,
        )
        early_stop_cb = EarlyStopCallback(
            metric_name="val_loss",
            threshold=0.5,
            target_epoch=3,
        )

        trainer = pl.Trainer(
            logger=wandb_logger,
            log_every_n_steps=10,
            accelerator=hp["ACCELERATOR"],
            devices=hp["DEVICES"],
            precision=hp["PRECISION"],
            max_epochs=hp["MAX_EPOCHS"],
            callbacks=[TQDMProgressBar(leave=True), save_model_cb, epoch_cb, early_stop_cb],
        )

        # Treinamento
        trainer.fit(model, dm)

        # Carrega melhor checkpoint
        best_model = CustomVectorModel.load_from_checkpoint(ckpt_path)

        # Teste
        trainer.test(best_model, dm)

        # Limpeza: apagar checkpoints
        ckpt_dir = os.path.dirname(ckpt_path)
        if os.path.exists(ckpt_dir):
            for fname in os.listdir(ckpt_dir):
                fpath = os.path.join(ckpt_dir, fname)
                if os.path.isfile(fpath):
                    os.remove(fpath)
            print(f"Todos os arquivos foram removidos da pasta {ckpt_dir}.")

        empty_trash()

        # Opcional: remover pasta do projeto local do W&B (mesmo do base)
        project_dir = os.path.expanduser(hp["PROJECT"])
        if os.path.exists(project_dir):
            shutil.rmtree(project_dir)
            print(f"A pasta {project_dir} foi excluída com sucesso.")
        else:
            print(f"A pasta {project_dir} não existe e não foi excluída.")

        wandb.finish()


if __name__ == "__main__":
    wandb.login()
    hp = load_hyperparameters("config.yaml")
    set_random_seeds(42)

    # Sweep: replicando o range do script-base
    sweep_config = {
        "method": "bayes",
        "metric": {"name": "val_loss", "goal": "minimize"},
        "early_terminate": {"type": "hyperband", "min_iter": 6, "eta": 3},
        "parameters": {
            "learning_rate": {
                "distribution": "log_uniform_values",
                "min": 3e-5,
                "max": 3e-3
            },
            "weight_decay": {
                "distribution": "log_uniform_values",
                "min": 1e-6,
                "max": 1e-2
            },
            "optimizer_momentum": {
                "distribution": "uniform",
                "min": 0.85,
                "max": 0.99
            },
            "label_smoothing": {
                "distribution": "uniform",
                "min": 0.0,
                "max": 0.2
            },
            "layer_scale": {
                "distribution": "uniform",
                "min": 0.5,
                "max": 2.0
            },
            "mlp_vector_model_scale": {
                "values": [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
            },
            "batch_size": {
                "values": [64]
            },
            "drop_path_rate": {
                "value": 0.0
            }
        }
    }

    sweep_id = wandb.sweep(sweep_config, project=hp["PROJECT"])
    # Ajuste "count" conforme desejar (200 no script-base)
    wandb.agent(sweep_id, function=train_model, count=200)
    wandb.finish()
