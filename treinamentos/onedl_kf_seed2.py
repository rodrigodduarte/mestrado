# train_images_no_balance.py  — CustomModel + (CustomImageModule / CustomImageModule_kf) SEM balanceamento

import os, time, json, random, yaml
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint

# --- seus módulos
from model import CustomModel  # CNN/Transformer + MLP final
from dataset import CustomImageModule, CustomImageModule_kf  # vamos escolher em runtime

# ---------------- utilidades ----------------
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
                n += len(first)
        return n

# ---------------- datamodule (sem balanceamento) ----------------
def build_datamodule(h, k_splits: int, fold_idx: int):
    """
    Usa CustomImageModule quando não for K-Fold; caso K_FOLDS>1, usa CustomImageModule_kf.
    Nenhum sampler balanceado, apenas shuffle padrão.
    """
    dm_kwargs = dict(
        train_dir=h["TRAIN_DIR"],
        test_dir=h["TEST_DIR"],
        shape=h["SHAPE"],
        batch_size=h["BATCH_SIZE"],
        num_workers=h["NUM_WORKERS"],
    )
    if k_splits and int(k_splits) > 1:
        return CustomImageModule_kf(**dm_kwargs, n_splits=k_splits, fold_idx=fold_idx)
    else:
        return CustomImageModule(**dm_kwargs)

# ---------------- modelo (sem pesos de classe) ----------------
def build_model(h):
    # garantir que optimizer_momentum seja tupla (o seu CustomModel espera isso)
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

# ---------------- treino principal ----------------
def main():
    h = load_hparams("config.yaml")
    n_seeds = int(h.get("N_SEEDS", 1))
    k_splits = int(h.get("K_FOLDS", 1))

    run_dir = os.path.join("modelos_kf", f"{h['NAME_DATASET']}_{h['TMODEL']}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "hyperparams_used.json"), "w") as f:
        json.dump(h, f, indent=2, ensure_ascii=False)

    for seed in range(42, 42 + n_seeds):
        print(f"\n==================== SEED {seed} ====================")
        set_seeds(seed)
        seed_dir = os.path.join(run_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        # Se k_splits==1, fazemos um único "fold_0" (compatível com sua árvore de pastas)
        effective_folds = k_splits if k_splits > 1 else 1

        for fold in range(effective_folds):
            print(f"\n==================== Fold {fold+1}/{effective_folds} ====================")
            fold_dir = os.path.join(seed_dir, f"fold_{fold}")
            os.makedirs(fold_dir, exist_ok=True)

            # --- DataModule (SEM balanceamento) ---
            dm = build_datamodule(h, k_splits=k_splits, fold_idx=fold)
            dm.setup(stage="fit")  # CustomImageModule: split 80/20 | CustomImageModule_kf: split por KFold
            # (não há class_weights nem sampler aqui)

            # --- Modelo (SEM class weights) ---
            model = build_model(h)  # CustomModel padronizado

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

            # ---------- Treino ----------
            t0 = time.perf_counter()
            trainer.fit(model, dm)
            train_time_sec = time.perf_counter() - t0

            # ---------- Melhor checkpoint ----------
            best_model_path = ckpt_cb.best_model_path
            try:
                ckpt_data = torch.load(best_model_path, map_location="cpu")
                best_epoch = ckpt_data.get("epoch", None)
            except Exception:
                best_epoch = None
            print(f"➡️  Best epoch (seed {seed}, fold {fold}): {best_epoch}")

            # Recarrega para validar/testar
            model = CustomModel.load_from_checkpoint(best_model_path)

            # ---------- Validação ----------
            val_metrics = trainer.validate(model, datamodule=dm, verbose=False)[0]

            # ---------- Teste + tempos ----------
            try:
                dm.setup(stage="test")
            except Exception:
                pass
            test_dl = dm.test_dataloader()
            n_test = _len_dataloader_safe(test_dl)

            t1 = time.perf_counter()
            test_metrics = trainer.test(model, datamodule=dm, verbose=False)[0]
            test_elapsed = time.perf_counter() - t1

            inf_ms = (test_elapsed * 1000.0 / n_test) if n_test > 0 else float("nan")
            throughput = (n_test / test_elapsed) if test_elapsed > 0 else float("nan")

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

            # ---------- Registro por fold ----------
            out_txt = os.path.join(fold_dir, "resultados_fold.txt")
            with open(out_txt, "w") as f:
                f.write(f"Seed: {seed}\n")
                f.write(f"Fold: {fold}\n")
                f.write(f"best_epoch: {best_epoch}\n")
                f.write(f"train_time_sec: {train_time_sec:.6f}\n")
                f.write(f"test_time_sec: {test_elapsed:.6f}\n")
                f.write(f"test_inf_ms_per_sample: {inf_ms:.6f}\n")
                f.write(f"throughput_samples_per_sec: {throughput:.6f}\n")
                if max_gpu_mb is not None:
                    f.write(f"max_gpu_mem_mb: {max_gpu_mb:.2f}\n")
                if model_size_mb is not None:
                    f.write(f"best_checkpoint_size_mb: {model_size_mb:.2f}\n")
                f.write(f"val_metrics_json: {json.dumps(val_metrics, ensure_ascii=False)}\n")
                f.write(f"test_metrics_json: {json.dumps(test_metrics, ensure_ascii=False)}\n")
                # Nenhum campo de balanceamento é salvo (por design)

            del model
            torch.cuda.empty_cache()

    print("\n==================== Treinamento concluído ====================")

if __name__ == "__main__":
    main()
