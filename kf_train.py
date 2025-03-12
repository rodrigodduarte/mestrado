import os
import shutil
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
    best_checkpoint_path = None
    epochs_per_fold = hyperparams['MAX_EPOCHS'] // k_splits  
    
    
    with wandb.init(project=hyperparams["PROJECT"], config=config)
        config_sweep = wandb.config
        
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
            optimizer_momentum=(config_sweep.optimizer_momentum, 0.999),  # AdamW usa dois betas
            weight_decay=float(config_sweep.weight_decay),
            layer_scale=config_sweep.layer_scale,
            mlp_vector_model_scale=config_sweep.mlp_vector_model_scale)
        
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

            checkpoint_path = f"{hyperparams['CHECKPOINT_PATH']}/fold_{fold+1}.ckpt"
            callbacks = [
                TQDMProgressBar(leave=True),
                SaveBestOrLastModelCallback(checkpoint_path),
                EarlyStoppingAtSpecificEpoch(patience=4, threshold=1e-3, monitor="val_loss"),
                EarlyStopCallback(metric_name="val_loss", threshold=0.7, target_epoch=4)
            ]

            wandb_logger = WandbLogger(project=hyperparams["PROJECT"], name=f"Fold_{fold+1}")

            trainer = pl.Trainer(
                logger=wandb_logger,
                log_every_n_steps=10,
                accelerator=hyperparams['ACCELERATOR'],
                devices=hyperparams['DEVICES'],
                precision=hyperparams['PRECISION'],
                max_epochs=epochs_per_fold,
                callbacks=callbacks
            )

            trainer.fit(model, data_module)
            
            best_checkpoint_path = checkpoint_path

        print(f"\nTreinamento finalizado. Melhor modelo salvo em: {best_checkpoint_path}")


        if best_checkpoint_path:
            print("\nIniciando teste final no melhor modelo...")
            best_model = CustomEnsembleModel.load_from_checkpoint(best_checkpoint_path)
            data_module.setup(stage='test')
            trainer.test(best_model, data_module)

            # 🔹 Definir diretório de destino e salvar o modelo diretamente lá
            final_model_dir = f"{hyperparams['PROJECT']}/runs/{wandb.run.name}"
            os.makedirs(final_model_dir, exist_ok=True)
            final_model_path = os.path.join(final_model_dir, "best_model.ckpt")

            # 🔹 Salvar o modelo final no diretório correto
            trainer.save_checkpoint(final_model_path)
            print(f"Melhor modelo salvo em: {final_model_path}")

            # Excluir diretório de checkpoints antigos
            if os.path.exists(hyperparams['CHECKPOINT_PATH']):
                shutil.rmtree(hyperparams['CHECKPOINT_PATH'])
                print(f"Diretório de checkpoints removido: {hyperparams['CHECKPOINT_PATH']}")
            else:
                print(f"Diretório {hyperparams['CHECKPOINT_PATH']} não encontrado, nada a remover.")



    wandb.finish()

if __name__ == "__main__":
    set_random_seeds()
    sweep_config = {
        'method': 'random',
        'metric': {'name': 'val_loss', 'goal': 'minimize'},
        'parameters': {
            'learning_rate': {'min': 1e-5, 'max': 2e-4, 'distribution': 'uniform'},
            'weight_decay': {'min': 1e-7, 'max': 1e-6, 'distribution': 'uniform'},
            'optimizer_momentum': {'min': 0.92, 'max': 0.99, 'distribution': 'uniform'},
            'mlp_vector_model_scale': {'min': 0.8, 'max': 1.3, 'distribution': 'uniform'},
            'layer_scale': {'min': 3, 'max': 4, 'distribution': 'uniform'},
            'drop_path_rate': {'min': 0.0, 'max': 0.5, 'distribution': 'uniform'},
            'label_smoothing': {'min': 0.0, 'max': 0.2, 'distribution': 'uniform'}
        }
    }
    sweep_id = wandb.sweep(sweep_config, project=load_hyperparameters('config.yaml')["PROJECT"])
    wandb.agent(sweep_id, function=train_model, count=200)
    wandb.finish()
