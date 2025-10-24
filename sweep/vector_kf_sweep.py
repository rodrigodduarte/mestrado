
# vector_kf_sweep.py — W&B sweep p/ CustomFeaturesOnlyModel + CustomFeaturesFromFoldersModule_kf (sem balanceamento)
# Referência: versão K-Fold original (vector_kf_seed1.py) adaptada para varredura de hiperparâmetros com Weights & Biases.

import os
import json
import yaml
import time
import random
import numpy as np
import torch
import pytorch_lightning as pl

from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from model import CustomFeaturesOnlyModel
from dataset import CustomFeaturesFromFoldersModule_kf

import wandb


# ---------------- utils ----------------
def _load_hparams():
    """Tenta carregar hparams de treinamentos/config1.yaml; se não existir, cai para config.yaml."""
    candidates = ["sweep/config1.yaml", "config.yaml"]
    for p in candidates:
        if os.path.exists(p):
            with open(p, "r") as f:
                return yaml.safe_load(f), p
    raise FileNotFoundError("Nenhum arquivo de config encontrado: sweep/config1.yaml ou config.yaml")

def _set_seeds(seed: int = 42):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def _infer_features_dim(dm):
    """Lê um batch (features,label) do train_dataloader e infere D das features."""
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    if isinstance(batch, (tuple, list)):
        if len(batch) == 2:
            feats, _ = batch
        elif len(batch) == 3:
            # caso o dataset retorne (img, feats, y) – por segurança
            _, feats, _ = batch
        else:
            raise RuntimeError("Batch inesperado (esperado 2 ou 3 elementos)")
    elif isinstance(batch, dict):
        feats = batch.get("features", None)
        if feats is None:
            raise RuntimeError("Batch dict esperado com chave 'features'.")
    else:
        raise RuntimeError("Batch inesperado")

    if not torch.is_tensor(feats):
        feats = torch.as_tensor(feats)
    if feats.ndim == 1:
        return int(feats.shape[0])
    return int(feats.shape[-1])

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
                try:
                    n += len(first)
                except Exception:
                    n += 1
        return n


# ---------------- treino por execução (usada pelo agent) ----------------
def train_once(config=None):
    h, hpath = _load_hparams()
    _set_seeds(42)

    with wandb.init(project=h["PROJECT"], config=config):
        cfg = wandb.config

        # K-Fold efetivo (se K_FOLDS==1, emula 5)
        k_cfg = int(h.get("K_FOLDS", 1))
        n_splits = k_cfg if k_cfg > 1 else 5
        fold_idx = int(getattr(cfg, "fold_idx", 0))

        # --- DataModule (sem balanceamento) ---
        dm = CustomFeaturesFromFoldersModule_kf(
            train_dir=h["TRAIN_DIR"],
            test_dir=h["TEST_DIR"],
            shape=h.get("SHAPE", (224, 224)),       # compat
            batch_size=h["BATCH_SIZE"],
            num_workers=h["NUM_WORKERS"],
            n_splits=n_splits,
            fold_idx=fold_idx,
            balance="none"
        )
        dm.setup(stage="fit")

        # Descobre número de classes se disponível
        num_classes = getattr(dm, "num_classes", h["NUM_CLASSES"])

        # features_dim: usa hparams se existir, senão infere do batch
        features_dim = int(h.get("FEATURES_DIM", 0)) or _infer_features_dim(dm)
        print(f"[info] features_dim={features_dim} | num_classes={num_classes} | fold={fold_idx}/{n_splits-1}")

        # --- Modelo ---
        betas = (float(cfg.optimizer_momentum), 0.999) if hasattr(cfg, "optimizer_momentum") else (0.9, 0.999)
        model = CustomFeaturesOnlyModel(
            name_dataset=h["NAME_DATASET"],
            shape=h.get("SHAPE", (224, 224)),
            epochs=h["MAX_EPOCHS"],
            learning_rate=float(cfg.learning_rate),
            features_dim=features_dim,
            drop_path_rate=h.get("DROP_PATH_RATE", 0.0),
            num_classes=num_classes,
            label_smoothing=float(cfg.label_smoothing),
            optimizer_momentum=betas,
            weight_decay=float(cfg.weight_decay),
            layer_scale=float(cfg.layer_scale),
        )

        # --- Logger + checkpoints ---
        wandb_logger = WandbLogger(project=h["PROJECT"])
        run_name = wandb.run.name
        ckpt_dir = h.get("CHECKPOINT_PATH", "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f"{run_name}.ckpt")

        ckpt_cb = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename=run_name,
            monitor="val_loss",
            mode="min",
            save_top_k=1
        )
        earlystop = EarlyStopping(monitor="val_loss", mode="min", patience=10)

        trainer = pl.Trainer(
            logger=wandb_logger,
            log_every_n_steps=10,
            accelerator=h["ACCELERATOR"],
            devices=h["DEVICES"],
            precision=h["PRECISION"],
            max_epochs=h["MAX_EPOCHS"],
            callbacks=[TQDMProgressBar(leave=True), ckpt_cb, earlystop],
        )

        # --- fit ---
        t0 = time.perf_counter()
        trainer.fit(model, dm)
        train_time_sec = time.perf_counter() - t0

        # --- best epoch (seguro) ---
        try:
            ckpt_meta = torch.load(ckpt_cb.best_model_path, map_location="cpu", weights_only=True)
        except TypeError:
            ckpt_meta = torch.load(ckpt_cb.best_model_path, map_location="cpu")
        best_epoch = ckpt_meta.get("epoch", None)
        print(f"[info] best epoch: {best_epoch}")

        # --- test ---
        try:
            dm.setup(stage="test")
        except Exception:
            pass
        best_model = CustomFeaturesOnlyModel.load_from_checkpoint(ckpt_cb.best_model_path)
        t1 = time.perf_counter()
        test_metrics = trainer.test(best_model, dm, verbose=False)[0]
        test_elapsed = time.perf_counter() - t1
        n_test = _len_dataloader_safe(dm.test_dataloader())
        wandb.log({
            "best_epoch": best_epoch if best_epoch is not None else -1,
            "train_time_sec": train_time_sec,
            "test_time_sec": test_elapsed,
            "test_inf_ms_per_sample": (test_elapsed * 1000.0 / n_test) if n_test > 0 else float("nan"),
            **{f"test_{k}": v for k, v in test_metrics.items()}
        })

        # opcional: limpar checkpoints antigos neste diretório
        # (mantemos o melhor por run)
        # for f in os.listdir(ckpt_dir):
        #     if f.endswith(".ckpt") and f != f"{run_name}.ckpt":
        #         try: os.remove(os.path.join(ckpt_dir, f))
        #         except: pass

        wandb.finish()


# ---------------- sweep launcher ----------------
if __name__ == "__main__":
    wandb.login()
    h, hpath = _load_hparams()
    _set_seeds(42)

    # folds a varrer (se K_FOLDS==1, usa 5)
    k_cfg = int(h.get("K_FOLDS", 1))
    effective_splits = k_cfg if k_cfg > 1 else 5

    sweep_config = {
        "method": "random",
        "metric": {"name": "val_loss", "goal": "minimize"},
        "parameters": {
            "learning_rate":     {"min": 1e-5, "max": 2e-4, "distribution": "uniform"},
            "weight_decay":      {"min": 1e-7, "max": 1e-5, "distribution": "uniform"},
            "optimizer_momentum":{"min": 0.90, "max": 0.99, "distribution": "uniform"},
            "layer_scale":       {"min": 0.5, "max": 4.0,  "distribution": "uniform"},
            "label_smoothing":   {"min": 0.0, "max": 0.2,  "distribution": "uniform"},
            "fold_idx":          {"values": list(range(effective_splits))}
        }
    }

    sweep_id = wandb.sweep(sweep_config, project=h["PROJECT"])
    wandb.agent(sweep_id, function=train_once, count=150)
    wandb.finish()
