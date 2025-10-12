import os
import shutil
import torch
import pytorch_lightning as pl
import numpy as np
import yaml
import random
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
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
def set_random_seeds(seed):
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Função principal para treinamento com validação cruzada
def train_model():
    hyperparams = load_hyperparameters('config2.yaml')
    k_splits = hyperparams['K_FOLDS']
    n_seeds = hyperparams.get('N_SEEDS', 1)  # padrão = 1 se não definido
    metrics_history = {}

    run_dir = os.path.join("modelos_kf", f"{hyperparams['NAME_DATASET']}_{hyperparams['TMODEL']}")
    os.makedirs(run_dir, exist_ok=True)

    for seed in range(42, 42 + n_seeds):
        print(f"\n==================== Treinando com SEED {seed} ====================")
        set_random_seeds(seed)

        # diretório da seed
        seed_dir = os.path.join(run_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        for fold in range(k_splits):
            print(f"\n==================== Fold {fold+1}/{k_splits} ====================")

            # diretório seed/fold
            fold_dir = os.path.join(seed_dir, f"fold_{fold}")
            os.makedirs(fold_dir, exist_ok=True)

            fold_callback = ModelCheckpoint(
                dirpath=fold_dir,                 # <-- agora salva em seed/fold
                filename="best_model",            # arquivo fica best_model.ckpt
                monitor="val_loss",
                mode="min",
                save_top_k=1
            )

            model = CustomEnsembleModel(
                tmodel=hyperparams["TMODEL"],
                name_dataset=hyperparams["NAME_DATASET"],
                shape=hyperparams["SHAPE"],
                epochs=hyperparams['MAX_EPOCHS'],
                learning_rate=hyperparams['LEARNING_RATE'],
                features_dim=hyperparams["FEATURES_DIM"],
                drop_path_rate=hyperparams['DROP_PATH_RATE'],
                num_classes=hyperparams['NUM_CLASSES'],
                label_smoothing=hyperparams['LABEL_SMOOTHING'],
                optimizer_momentum=(hyperparams['OPTIMIZER_MOMENTUM'], 0.999),
                weight_decay=hyperparams['WEIGHT_DECAY'],
                layer_scale=hyperparams['LAYER_SCALE']
            )

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
                log_every_n_steps=10,
                accelerator=hyperparams['ACCELERATOR'],
                devices=hyperparams['DEVICES'],
                precision=hyperparams['PRECISION'],
                max_epochs=hyperparams['MAX_EPOCHS'],
                callbacks=callbacks
            )

            trainer.fit(model, data_module)

            best_model_path = fold_callback.best_model_path

            # Novo trecho: carregar epoch do melhor modelo
            checkpoint_data = torch.load(best_model_path, map_location='cpu')
            best_epoch = checkpoint_data['epoch']
            print(f"Melhor modelo para seed {seed}, fold {fold} salvo na época {best_epoch}")

            model = CustomEnsembleModel.load_from_checkpoint(best_model_path)
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

            del model
            torch.cuda.empty_cache()

    print("\n==================== Métricas Finais ====================")
    for metric_name, values in metrics_history.items():
        if isinstance(values[0], (int, float, np.float32, np.float64)):
            mean = np.mean(values)
            std = np.std(values)
            print(f"{metric_name}: mean = {mean:.4f}, std = {std:.4f}")

if __name__ == "__main__":
    train_model()
