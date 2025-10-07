import os
import torch
import pytorch_lightning as pl
import numpy as np

from pytorch_lightning.callbacks import TQDMProgressBar

# ==== ALTERAÇÕES PRINCIPAIS ====
# 1) Modelo passa a ser o de FEATURES-ONLY
from model import CustomFeaturesOnlyModel
# 2) DataModule passa a ler SOMENTE VETORES via pastas por classe
#    (mesma assinatura do CustomImageCSVModule, ignorando 'shape')
from dataset import CustomFeaturesFromFoldersModule

from callbacks import EarlyStoppingAtSpecificEpoch, SaveBestOrLastModelCallback, EarlyStopCallback

import yaml
import wandb
from pytorch_lightning.loggers import WandbLogger
import random
import subprocess
import shutil


def empty_trash():
    trash_path = os.path.expanduser("~/.local/share/Trash")
    if os.path.exists(trash_path):
        subprocess.run(["rm", "-rf", f"{trash_path}/files/*", f"{trash_path}/info/*"]) \
            and print("Lixeira esvaziada com sucesso no Linux.")


def load_hyperparameters(file_path):
    with open(file_path, 'r') as file:
        hyperparams = yaml.safe_load(file)
    return hyperparams


def set_random_seeds(seed: int = 42):
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ================================
# Treino (PIPELINE DO SWEEP) — FEATURES ONLY, SEM K-FOLD
# ================================

def train_model(config=None):
    hyperparams = load_hyperparameters('config.yaml')

    with wandb.init(project=hyperparams["PROJECT"], config=config):
        config_sweep = wandb.config

        # ----------------------------
        # DataModule (features em PASTAS por classe)
        # Mantém a mesma assinatura do CustomImageCSVModule: 'shape' é aceito e ignorado
        # Split de validação interno (val_split) fixo em 0.2 por padrão
        # ----------------------------
        data_module = CustomFeaturesFromFoldersModule(
            train_dir=hyperparams['TRAIN_DIR'],
            test_dir=hyperparams['TEST_DIR'],
            batch_size=hyperparams['BATCH_SIZE'],
            num_workers=hyperparams['NUM_WORKERS'],
            val_split=float(hyperparams.get('VAL_SPLIT', 0.2)),
            seed=int(hyperparams.get('SEED', 42)),
        )

        # ----------------------------
        # Modelo (features only)
        # Mantém nomes/assinatura compatíveis com seu YAML / molde original
        # ----------------------------
        model = CustomFeaturesOnlyModel(
            tmodel=hyperparams.get("TMODEL", "vector_only"),
            name_dataset=hyperparams.get("NAME_DATASET", "features"),
            shape=hyperparams.get("SHAPE", (0, 0)),              # aceito/ignorado
            epochs=int(hyperparams['MAX_EPOCHS']),
            learning_rate=float(config_sweep.learning_rate),
            features_dim=int(hyperparams['FEATURES_DIM']),
            drop_path_rate=float(config_sweep.get('drop_path_rate', 0.0)),  # aceito/ignorado
            num_classes=int(hyperparams['NUM_CLASSES']),
            label_smoothing=float(config_sweep.label_smoothing),
            optimizer_momentum=(float(config_sweep.optimizer_momentum), 0.999),
            weight_decay=float(config_sweep.weight_decay),
            layer_scale=float(config_sweep.layer_scale),
        )

        # ----------------------------
        # Logger e callbacks
        # ----------------------------
        wandb_logger = WandbLogger(project=hyperparams["PROJECT"]) 
        run_name = wandb.run.name
        checkpoint_path = f"{hyperparams['CHECKPOINT_PATH']}/{run_name}.ckpt"
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        save_model_callback = SaveBestOrLastModelCallback(checkpoint_path)

        epoch_callback = EarlyStoppingAtSpecificEpoch(
            patience=2,
            threshold=1e-3,
            monitor="val_loss",
            mode="min",
            verbose=True
        )

        early_stop_callback = EarlyStopCallback(
            metric_name="val_loss",
            threshold=0.5,
            target_epoch=3
        )

        # ----------------------------
        # Trainer
        # ----------------------------
        trainer = pl.Trainer(
            logger=wandb_logger,
            log_every_n_steps=10,
            accelerator=hyperparams['ACCELERATOR'],
            devices=hyperparams['DEVICES'],
            precision=hyperparams['PRECISION'],
            max_epochs=int(hyperparams['MAX_EPOCHS']),
            callbacks=[
                TQDMProgressBar(leave=True),
                save_model_callback,
                epoch_callback,
                early_stop_callback
            ]
        )

        # ----------------------------
        # Fit + Test
        # ----------------------------
        trainer.fit(model, data_module)

        # Carrega melhor checkpoint salvo
        model = CustomFeaturesOnlyModel.load_from_checkpoint(checkpoint_path)
        trainer.test(model, data_module)

        # ----------------------------
        # Limpeza (mesmo molde)
        # ----------------------------
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if os.path.exists(checkpoint_dir):
            for file_name in os.listdir(checkpoint_dir):
                file_path = os.path.join(checkpoint_dir, file_name)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            print(f"Todos os arquivos foram removidos da pasta {checkpoint_dir}.")

        empty_trash()

        # Atenção: no molde original havia remoção de uma pasta com o nome do PROJECT.
        project_dir = os.path.expanduser(hyperparams["PROJECT"])  # mantenho o comportamento original
        if os.path.exists(project_dir):
            shutil.rmtree(project_dir)
            print(f"A pasta {project_dir} foi excluída com sucesso.")
        else:
            print(f"A pasta {project_dir} não existe e não foi excluída.")  

        wandb.finish()


if __name__ == "__main__":
    # Login no W&B
    wandb.login()
    hyperparams = load_hyperparameters('config.yaml')

    # Seeds
    set_random_seeds(int(hyperparams.get('SEED', 42)))

    # Sweep config — mantém faixas do seu molde
    sweep_config = {
        'method': 'random',
        'metric': {'name': 'val_loss', 'goal': 'minimize'},
        'parameters': {
            'learning_rate':      {'min': 1e-5, 'max': 2e-4, 'distribution': 'uniform'},
            'weight_decay':       {'min': 1e-7, 'max': 1e-6, 'distribution': 'uniform'},
            'optimizer_momentum': {'min': 0.92, 'max': 0.99, 'distribution': 'uniform'},
            'mlp_vector_model_scale': {'min': 0.8, 'max': 1.3, 'distribution': 'uniform'},  # ignorado aqui
            'layer_scale':        {'min': 0.5, 'max': 1.5, 'distribution': 'uniform'},
            'drop_path_rate':     {'min': 0.0, 'max': 0.5, 'distribution': 'uniform'},      # aceito/ignorado
            'label_smoothing':    {'min': 0.0, 'max': 0.2, 'distribution': 'uniform'}
        }
    }

    # Criar o sweep
    sweep_id = wandb.sweep(sweep_config, project=hyperparams["PROJECT"])

    # Executar o sweep (sem K-Fold)
    wandb.agent(sweep_id, function=train_model, count=hyperparams.get('SWEEP_COUNT', 200))

    wandb.finish()
