# train_kf_db_features.py — Pipeline K-Fold dedicado a modelos de FEATURES (SSN)
# Mantém a “cara” do seu script original e só troca o modelo/datamodule e integra pesos de classe.

import os
import json
import time
import random
from typing import Any, Dict, Optional

import yaml
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint

# ==== IMPORTS ESPECÍFICOS DOS NOVOS COMPONENTES (apenas isto muda) ====
# Use as classes que você já tem nos arquivos novos:
#   - CustomFeaturesOnlyModel.py
#   - CustomFeaturesFromFoldersModule.py   (versão K-Fold)
from model import CustomFeaturesOnlyModel
from dataset import CustomFeaturesFromFoldersModule_kf


# ============================ Utils ============================

def load_hparams(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def set_seeds(seed: int):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _len_dataloader_safe(dl) -> int:
    try:
        return len(dl.dataset)
    except Exception:
        n = 0
        for b in dl:
            first = b[0] if isinstance(b, (list, tuple)) else b
            try:
                n += first.size(0)
            except Exception:
                try:
                    n += len(first)
                except Exception:
                    n += 1
        return n

def _maybe_get_class_weights(dm) -> Optional[torch.Tensor]:
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


# ============================ Treino ============================

def train():
    h = load_hparams("config.yaml")

    NAME_DATASET = h["NAME_DATASET"]
    TMODEL      = h.get("TMODEL", "features_mlp")  # mantemos campo, só para nome do run
    TRAIN_DIR   = h["TRAIN_DIR"]
    TEST_DIR    = h["TEST_DIR"]
    SHAPE       = tuple(h.get("SHAPE", [224, 224, 3]))  # ignorado pelo datamodule de features, mantido por compat.
    NUM_CLASSES = int(h["NUM_CLASSES"])
    FEATURES_DIM= int(h.get("FEATURES_DIM", 648))

    K_FOLDS     = int(h["K_FOLDS"])
    MAX_EPOCHS  = int(h["MAX_EPOCHS"])
    BATCH_SIZE  = int(h["BATCH_SIZE"])
    NUM_WORKERS = int(h.get("NUM_WORKERS", 4))
    BASE_SEED   = int(h.get("BASE_SEED", 42))
    N_SEEDS     = int(h.get("N_SEEDS", 1))

    LEARNING_RATE     = float(h.get("LEARNING_RATE", 1e-3))
    WEIGHT_DECAY      = float(h.get("WEIGHT_DECAY", 1e-4))
    LABEL_SMOOTHING   = float(h.get("LABEL_SMOOTHING", 0.0))
    OPTIMIZER_MOMENTUM= h.get("OPTIMIZER_MOMENTUM", (0.9, 0.999))
    LAYER_SCALE       = float(h.get("LAYER_SCALE", 1.0))
    DROP_PATH_RATE    = float(h.get("DROP_PATH_RATE", 0.0))  # compat.
    BALANCE_MODE      = h.get("BALANCE_MODE", "none")        # 'none'|'sampler'|'weights'|'both'

    # Precisão: para features/SSN, 32 bits costuma ser mais estável
    PRECISION    = h.get("PRECISION", 32)
    ACCELERATOR  = h.get("ACCELERATOR", "gpu" if torch.cuda.is_available() else "cpu")
    DEVICES      = h.get("DEVICES", 1)

    # Diretório de resultados (mesmo padrão do original)
    run_dir = os.path.join("modelos_kf", f"{NAME_DATASET}_{TMODEL}")
    os.makedirs(run_dir, exist_ok=True)

    # Guarda os hparams usados
    with open(os.path.join(run_dir, "hyperparams_used.json"), "w") as f:
        json.dump(h, f, indent=2, ensure_ascii=False)

    for s in range(BASE_SEED, BASE_SEED + N_SEEDS):
        print(f"\n==================== Treinando com SEED {s} ====================")
        set_seeds(s)
        seed_dir = os.path.join(run_dir, f"seed_{s}")
        os.makedirs(seed_dir, exist_ok=True)

        for fold in range(K_FOLDS):
            print(f"\n==================== Fold {fold+1}/{K_FOLDS} ====================")
            fold_dir = os.path.join(seed_dir, f"fold_{fold}")
            os.makedirs(fold_dir, exist_ok=True)

            # --------- DataModule (features) ---------
            dm = CustomFeaturesFromFoldersModule_kf(
                train_dir=TRAIN_DIR,
                test_dir=TEST_DIR,
                shape=SHAPE,                   # mantido por compat.
                batch_size=BATCH_SIZE,
                num_workers=NUM_WORKERS,
                n_splits=K_FOLDS,
                fold_idx=fold,
                balance=BALANCE_MODE          # 'none'|'sampler'|'weights'|'both'
            )
            dm.setup(stage="fit")
            class_weights = _maybe_get_class_weights(dm)

            # --------- Modelo (features) --------------
            model = CustomFeaturesOnlyModel(
                name_dataset=NAME_DATASET,
                shape=SHAPE,
                epochs=MAX_EPOCHS,
                learning_rate=LEARNING_RATE,
                features_dim=FEATURES_DIM,    # <<< garante 648 (ou o que estiver no YAML)
                drop_path_rate=DROP_PATH_RATE,
                num_classes=NUM_CLASSES,
                label_smoothing=LABEL_SMOOTHING,
                optimizer_momentum=OPTIMIZER_MOMENTUM,
                weight_decay=WEIGHT_DECAY,
                layer_scale=LAYER_SCALE,
                class_weights=class_weights
            )

            # --------- Callbacks ----------
            ckpt_cb = ModelCheckpoint(
                dirpath=fold_dir,
                filename="best_model",
                monitor="val_loss",
                mode="min",
                save_top_k=1
            )
            callbacks = [TQDMProgressBar(leave=True), ckpt_cb]

            # --------- Trainer ------------
            trainer = pl.Trainer(
                log_every_n_steps=10,
                accelerator=ACCELERATOR,
                devices=DEVICES,
                precision=PRECISION,           # 32 por padrão (features-only)
                max_epochs=MAX_EPOCHS,
                callbacks=callbacks
            )

            # --------- Fit ----------------
            t0 = time.perf_counter()
            trainer.fit(model, dm)
            train_time_sec = time.perf_counter() - t0

            # Carrega melhor
            best_model_path = ckpt_cb.best_model_path
            best_epoch = None
            try:
                ckpt = torch.load(best_model_path, map_location="cpu")
                best_epoch = ckpt.get("epoch", None)
            except Exception:
                pass
            print(f"Melhor modelo (seed {s}, fold {fold}) na época: {best_epoch}")

            try:
                model = CustomFeaturesOnlyModel.load_from_checkpoint(best_model_path)
            except Exception:
                pass

            # --------- Validate/Test ------
            try:
                dm.setup(stage="test")
            except Exception:
                pass

            val_metrics = trainer.validate(model, datamodule=dm, verbose=False)
            val_metrics = val_metrics[0] if len(val_metrics) else {}

            test_dl = dm.test_dataloader()
            n_test = _len_dataloader_safe(test_dl)

            t1 = time.perf_counter()
            test_metrics = trainer.test(model, datamodule=dm, verbose=False)
            test_metrics = test_metrics[0] if len(test_metrics) else {}
            test_elapsed_sec = time.perf_counter() - t1

            inf_ms = (test_elapsed_sec * 1000.0 / n_test) if n_test > 0 else float("nan")
            throughput = (n_test / test_elapsed_sec) if test_elapsed_sec > 0 else float("nan")

            max_gpu_mem_mb = None
            if torch.cuda.is_available():
                try:
                    max_gpu_mem_mb = torch.cuda.max_memory_allocated() / 1e6
                except Exception:
                    pass

            model_size_mb = None
            try:
                model_size_mb = os.path.getsize(best_model_path) / 1e6
            except Exception:
                pass

            # --------- Registro por fold ---
            fold_txt = os.path.join(fold_dir, "resultados_fold.txt")
            with open(fold_txt, "w") as f:
                f.write(f"Seed: {s}\n")
                f.write(f"Fold: {fold}\n")
                f.write(f"best_epoch: {best_epoch}\n")
                f.write(f"train_time_sec: {train_time_sec:.6f}\n")
                f.write(f"test_time_sec: {test_elapsed_sec:.6f}\n")
                f.write(f"test_inf_ms_per_sample: {inf_ms:.6f}\n")
                f.write(f"throughput_samples_per_sec: {throughput:.6f}\n")
                if max_gpu_mem_mb is not None:
                    f.write(f"max_gpu_mem_mb: {max_gpu_mem_mb:.2f}\n")
                if model_size_mb is not None:
                    f.write(f"best_checkpoint_size_mb: {model_size_mb:.2f}\n")
                f.write(f"val_metrics_json: {json.dumps(val_metrics, ensure_ascii=False)}\n")
                f.write(f"test_metrics_json: {json.dumps(test_metrics, ensure_ascii=False)}\n")
                f.write("pipeline: features\n")
                f.write(f"balance_mode: {BALANCE_MODE}\n")
                if class_weights is not None:
                    try:
                        cw = class_weights.detach().cpu().numpy()
                    except Exception:
                        try:
                            cw = np.array(class_weights)
                        except Exception:
                            cw = None
                    if cw is not None:
                        f.write(f"class_weights: {np.round(cw, 6).tolist()}\n")

            del model
            torch.cuda.empty_cache()

    print("\n==================== Treinamento (features) concluído ====================")


if __name__ == "__main__":
    train()
