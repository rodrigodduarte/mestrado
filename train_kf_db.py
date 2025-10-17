# train_kf_no_wandb_f.py  — versão adaptada p/ CustomModel + CustomImageModule_kf (desbalanceamento)

import os
import time
import json
import torch
import pytorch_lightning as pl
import numpy as np
import yaml
import random
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint

# ====== IMPORTS (ajuste o caminho se necessário) ======
from model import CustomModel
from dataset import CustomImageModule_kf

# (se você usa callbacks próprios, mantenha)
from callbacks import (
    EarlyStoppingAtSpecificEpoch,
    SaveBestOrLastModelCallback,
    EarlyStopCallback
)

# ---------------- utilidades ----------------
def load_hyperparameters(file_path):
    with open(file_path, 'r') as file:
        hyperparams = yaml.safe_load(file)
    return hyperparams

def set_random_seeds(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _len_dataloader_safe(dl):
    try:
        return len(dl.dataset)
    except Exception:
        n = 0
        for b in dl:
            first = b[0] if isinstance(b, (list, tuple)) else b
            try:
                n += first.size(0)
            except Exception:
                n += len(first)
        return n

def _maybe_get_class_weights(dm):
    """
    Pega pesos de classe do datamodule se disponível.
    Prioriza método get_class_weights(); cai para atributo class_weights.
    """
    cw = None
    if hasattr(dm, "get_class_weights"):
        try:
            cw = dm.get_class_weights()
        except Exception:
            cw = None
    if cw is None and hasattr(dm, "class_weights"):
        try:
            cw = dm.class_weights
        except Exception:
            cw = None
    return cw

def _build_datamodule(hparams, k_splits, fold):
    # Tenta passar 'balance' se o DataModule aceitar; caso contrário, ignora.
    dm_kwargs = dict(
        train_dir=hparams['TRAIN_DIR'],
        test_dir=hparams['TEST_DIR'],
        shape=hparams['SHAPE'],
        batch_size=hparams['BATCH_SIZE'],
        num_workers=hparams['NUM_WORKERS'],
        n_splits=k_splits,
        fold_idx=fold
    )
    balance_mode = hparams.get('BALANCE_MODE', 'none')
    try:
        dm = CustomImageModule_kf(**dm_kwargs, balance=balance_mode)
    except TypeError:
        dm = CustomImageModule_kf(**dm_kwargs)  # versões antigas sem 'balance'
        balance_mode = 'none'
    return dm, balance_mode

def _build_model(hparams, class_weights):
    """
    Constrói CustomModel, injetando class_weights no ctor.
    Se o ctor não aceitar, tenta set_class_weights (retrocompatibilidade).
    """
    model_args = dict(
        tmodel=hparams["TMODEL"],
        name_dataset=hparams["NAME_DATASET"],
        shape=hparams["SHAPE"],
        epochs=hparams['MAX_EPOCHS'],
        learning_rate=hparams['LEARNING_RATE'],
        drop_path_rate=hparams['DROP_PATH_RATE'],
        num_classes=hparams['NUM_CLASSES'],
        label_smoothing=hparams['LABEL_SMOOTHING'],
        optimizer_momentum=(hparams['OPTIMIZER_MOMENTUM'], 0.999),
        weight_decay=hparams['WEIGHT_DECAY'],
        layer_scale=hparams['LAYER_SCALE']
    )

    try:
        model = CustomModel(**model_args, class_weights=class_weights)
        return model, True  # passou via ctor
    except TypeError:
        model = CustomModel(**model_args)
        injected = False
        try:
            if class_weights is not None and hasattr(model, "set_class_weights"):
                model.set_class_weights(class_weights)
                injected = True
        except Exception:
            injected = False
        return model, injected

# ---------------- treino principal ----------------
def train_model():
    h = load_hyperparameters('config2.yaml')
    k_splits = h['K_FOLDS']
    n_seeds = h.get('N_SEEDS', 1)

    run_dir = os.path.join("modelos_kf", f"{h['NAME_DATASET']}_{h['TMODEL']}")
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "hyperparams_used.json"), "w") as f:
        json.dump(h, f, indent=2, ensure_ascii=False)

    for seed in range(42, 42 + n_seeds):
        print(f"\n==================== Treinando com SEED {seed} ====================")
        set_random_seeds(seed)

        seed_dir = os.path.join(run_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        for fold in range(k_splits):
            print(f"\n==================== Fold {fold+1}/{k_splits} ====================")
            fold_dir = os.path.join(seed_dir, f"fold_{fold}")
            os.makedirs(fold_dir, exist_ok=True)

            # --- DataModule primeiro (para obter class_weights/sampler) ---
            data_module, balance_mode = _build_datamodule(h, k_splits, fold)
            data_module.setup(stage='fit')
            class_weights = _maybe_get_class_weights(data_module)

            # --- Modelo (tenta injetar pesos de classe) ---
            model, cw_injected = _build_model(h, class_weights)

            fold_callback = ModelCheckpoint(
                dirpath=fold_dir,
                filename="best_model",
                monitor="val_loss",
                mode="min",
                save_top_k=1
            )

            callbacks = [TQDMProgressBar(leave=True), fold_callback]
            # (adicione seus callbacks próprios se precisar)
            # callbacks += [EarlyStopCallback(...), SaveBestOrLastModelCallback(...), ...]

            trainer = pl.Trainer(
                log_every_n_steps=10,
                accelerator=h['ACCELERATOR'],
                devices=h['DEVICES'],
                precision=h['PRECISION'],
                max_epochs=h['MAX_EPOCHS'],
                callbacks=callbacks
            )

            # ---------- Treino ----------
            t0 = time.perf_counter()
            trainer.fit(model, data_module)
            train_time_sec = time.perf_counter() - t0

            # ---------- Melhor checkpoint ----------
            best_model_path = fold_callback.best_model_path
            checkpoint_data = torch.load(best_model_path, map_location='cpu')
            best_epoch = checkpoint_data.get('epoch', None)
            print(f"Melhor modelo para seed {seed}, fold {fold} salvo na época {best_epoch}")

            model = CustomModel.load_from_checkpoint(best_model_path)

            # Reaproveita datamodule para teste
            try:
                data_module.setup(stage='test')
            except Exception:
                pass

            # ---------- Validação ----------
            val_metrics = trainer.validate(model, datamodule=data_module, verbose=False)[0]

            # ---------- Teste + tempos ----------
            test_dl = data_module.test_dataloader()
            num_test_samples = _len_dataloader_safe(test_dl)

            t1 = time.perf_counter()
            test_metrics = trainer.test(model, datamodule=data_module, verbose=False)[0]
            test_elapsed_sec = time.perf_counter() - t1

            test_inf_ms_per_sample = (test_elapsed_sec * 1000.0 / num_test_samples) if num_test_samples > 0 else float('nan')
            test_throughput = (num_test_samples / test_elapsed_sec) if test_elapsed_sec > 0 else float('nan')

            max_gpu_mem_mb = None
            if torch.cuda.is_available():
                try:
                    max_gpu_mem_mb = torch.cuda.max_memory_allocated() / 1e6
                except Exception:
                    max_gpu_mem_mb = None

            try:
                model_size_mb = os.path.getsize(best_model_path) / 1e6
            except Exception:
                model_size_mb = None

            # ---------- Registro por fold ----------
            fold_txt_path = os.path.join(fold_dir, "resultados_fold.txt")
            with open(fold_txt_path, "w") as f:
                f.write(f"Seed: {seed}\n")
                f.write(f"Fold: {fold}\n")
                f.write(f"best_epoch: {best_epoch}\n")
                f.write(f"train_time_sec: {train_time_sec:.6f}\n")
                f.write(f"test_time_sec: {test_elapsed_sec:.6f}\n")
                f.write(f"test_inf_ms_per_sample: {test_inf_ms_per_sample:.6f}\n")
                f.write(f"throughput_samples_per_sec: {test_throughput:.6f}\n")
                if max_gpu_mem_mb is not None:
                    f.write(f"max_gpu_mem_mb: {max_gpu_mem_mb:.2f}\n")
                if model_size_mb is not None:
                    f.write(f"best_checkpoint_size_mb: {model_size_mb:.2f}\n")
                f.write(f"val_metrics_json: {json.dumps(val_metrics, ensure_ascii=False)}\n")
                f.write(f"test_metrics_json: {json.dumps(test_metrics, ensure_ascii=False)}\n")
                # ---- novos campos (balanceamento) ----
                f.write(f"balance_mode: {balance_mode}\n")
                if class_weights is not None:
                    try:
                        f.write(f"class_weights: {np.round(class_weights.detach().cpu().numpy(), 6).tolist()}\n")
                    except Exception:
                        try:
                            f.write(f"class_weights: {np.round(np.array(class_weights), 6).tolist()}\n")
                        except Exception:
                            f.write(f"class_weights: <unavailable>\n")
                f.write(f"class_weights_injected: {bool(class_weights is not None and cw_injected)}\n")

            del model
            torch.cuda.empty_cache()

    print("\n==================== Treinamento concluído ====================")

if __name__ == "__main__":
    train_model()
