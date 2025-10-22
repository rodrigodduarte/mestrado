
# train_images_kf_only.py — Usa SEMPRE CustomImageModule_kf (sem balanceamento de classes)
# - Se K_FOLDS > 1: faz K-fold normalmente.
# - Se K_FOLDS == 1: usa n_splits=5 e executa apenas fold_idx=0 (≈ 80/20), mantendo a mesma classe _kf.

import os, time, json, random, yaml
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint

from model import CustomModel
from dataset import CustomImageModule_kf  # sempre esta classe

# ---------- utilidades ----------
def load_hparams(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def set_seeds(seed: int):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

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

def build_datamodule_kf(h, n_splits: int, fold_idx: int):
    """Sempre usa CustomImageModule_kf sem qualquer balanceamento de classes."""
    return CustomImageModule_kf(
        train_dir=h["TRAIN_DIR"],
        test_dir=h["TEST_DIR"],
        shape=h["SHAPE"],
        batch_size=h["BATCH_SIZE"],
        num_workers=h["NUM_WORKERS"],
        n_splits=n_splits,
        fold_idx=fold_idx,
    )

def build_model(h):
    # garantir que optimizer_momentum seja tupla
    opt_mom = h["OPTIMIZER_MOMENTUM"]
    if isinstance(opt_mom, (int, float)):
        opt_mom = (opt_mom, 0.999)
    elif isinstance(opt_mom, list):
        opt_mom = tuple(opt_mom)

    model = CustomModel(
        tmodel=h["TMODEL"],
        name_dataset=h["NAME_DATASET"],
        shape=h["SHAPE"],
        epochs=h["MAX_EPOCHS"],
        learning_rate=h["LEARNING_RATE"],
        drop_path_rate=h["DROP_PATH_RATE"],
        num_classes=h["NUM_CLASSES"],
        label_smoothing=h["LABEL_SMOOTHING"],
        optimizer_momentum=opt_mom,
        weight_decay=h["WEIGHT_DECAY"],
        layer_scale=h["LAYER_SCALE"],
    )
    return model

def main():
    h = load_hparams("config2.yaml")
    n_seeds = int(h.get("N_SEEDS", 1))
    k_splits_cfg = int(h.get("K_FOLDS", 1))

    # Estratégia "apenas _kf":
    # - Se K_FOLDS > 1: usar exatamente esse número de folds
    # - Se K_FOLDS == 1: emular split 80/20 usando n_splits=5 e rodar só fold_idx=0
    if k_splits_cfg > 1:
        effective_n_splits = k_splits_cfg
        effective_folds_to_run = list(range(effective_n_splits))
    else:
        effective_n_splits = 5
        effective_folds_to_run = [0]  # roda um único fold

    run_dir = os.path.join("modelos_kf", f"{h['NAME_DATASET']}_{h['TMODEL']}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "hyperparams_used.json"), "w") as f:
        json.dump(h, f, indent=2, ensure_ascii=False)

    for seed in range(42, 42 + n_seeds):
        print(f"==================== SEED {seed} ====================")
        set_seeds(seed)
        seed_dir = os.path.join(run_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        for fold_idx in effective_folds_to_run:
            print(f"\n==================== Fold {fold_idx+1}/{effective_n_splits} ====================")
            fold_dir = os.path.join(seed_dir, f"fold_{fold_idx}")
            os.makedirs(fold_dir, exist_ok=True)

            # DataModule (SEM balanceamento; apenas shuffle no treino)
            dm = build_datamodule_kf(h, n_splits=effective_n_splits, fold_idx=fold_idx)
            dm.setup(stage="fit")

            # Modelo
            model = build_model(h)

            ckpt_cb = ModelCheckpoint(
                dirpath=fold_dir,
                filename="best_model",
                monitor="val_loss",
                mode="min",
                save_top_k=1,
            )
            callbacks = [TQDMProgressBar(leave=True), ckpt_cb]

            trainer = pl.Trainer(
                log_every_n_steps=10,
                accelerator=h["ACCELERATOR"],
                devices=h["DEVICES"],
                precision=h["PRECISION"],
                max_epochs=h["MAX_EPOCHS"],
                callbacks=callbacks,
            )

            # --- Treino ---
            t0 = time.perf_counter()
            trainer.fit(model, dm)
            train_time_sec = time.perf_counter() - t0

            # Melhor checkpoint
            best_model_path = ckpt_cb.best_model_path
            try:
                ckpt_data = torch.load(best_model_path, map_location="cpu")
                best_epoch = ckpt_data.get("epoch", None)
            except Exception:
                best_epoch = None
            print(f"➡️  Best epoch (seed {seed}, fold {fold_idx}): {best_epoch}")

            # Recarrega e valida/testa
            model = CustomModel.load_from_checkpoint(best_model_path)

            val_metrics = trainer.validate(model, datamodule=dm, verbose=False)[0]

            # Teste
            try:
                dm.setup(stage="test")
            except Exception:
                pass
            test_dl = dm.test_dataloader()
            n_test = _len_dataloader_safe(test_dl)

            t1 = time.perf_counter()
            test_metrics = trainer.test(model, datamodule=dm, verbose=False)[0]
            test_time_sec = time.perf_counter() - t1

            inf_ms = (test_time_sec * 1000.0 / n_test) if n_test > 0 else float("nan")
            throughput = (n_test / test_time_sec) if test_time_sec > 0 else float("nan")

            max_gpu_mb = None
            if torch.cuda.is_available():
                try:
                    max_gpu_mb = torch.cuda.max_memory_allocated() / 1e6
                except Exception:
                    pass

            model_size_mb = None
            try:
                model_size_mb = os.path.getsize(best_model_path) / 1e6
            except Exception:
                pass

            # Registro por fold
            out_txt = os.path.join(fold_dir, "resultados_fold.txt")
            with open(out_txt, "w") as f:
                f.write(f"Seed: {seed}\n")
                f.write(f"Fold: {fold_idx}\n")
                f.write(f"n_splits: {effective_n_splits}\n")
                f.write(f"best_epoch: {best_epoch}\n")
                f.write(f"train_time_sec: {train_time_sec:.6f}\n")
                f.write(f"test_time_sec: {test_time_sec:.6f}\n")
                f.write(f"test_inf_ms_per_sample: {inf_ms:.6f}\n")
                f.write(f"throughput_samples_per_sec: {throughput:.6f}\n")
                if max_gpu_mb is not None:
                    f.write(f"max_gpu_mem_mb: {max_gpu_mb:.2f}\n")
                if model_size_mb is not None:
                    f.write(f"best_checkpoint_size_mb: {model_size_mb:.2f}\n")
                f.write(f"val_metrics_json: {json.dumps(val_metrics, ensure_ascii=False)}\n")
                f.write(f"test_metrics_json: {json.dumps(test_metrics, ensure_ascii=False)}\n")

            del model
            torch.cuda.empty_cache()

    print("\n==================== Concluído ====================")

if __name__ == "__main__":
    main()