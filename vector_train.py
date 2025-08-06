#!/usr/bin/env python3
"""
kf_train_vector.py
──────────────────
Treina um **MLP de duas camadas que usa apenas o vetor de características**,
mas mantém exatamente a mesma interface/hiper-parâmetros do pipeline
`modelotriplo_train.py`.

• Modelo  : CustomVectorModel  (novo – só vetor)
• DataMod : CustomImageCSVModule_kf (fornece (img, features, label);
           as imagens são ignoradas pelo modelo)
• Checkpts: modelos_kf/<DATASET>_vector/fold_<k>_best_model.ckpt
"""

# ------------------------------------------------------------------ #
# imports padrão
# ------------------------------------------------------------------ #
import os, random, yaml, torch, pytorch_lightning as pl, numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar

# projeto
from model import ReLuMLP2L                # ← novo modelo
from kf_data import CustomImageCSVModule_kf
from callbacks import EarlyStoppingAtSpecificEpoch

# ------------------------------------------------------------------ #
# helpers
# ------------------------------------------------------------------ #
def load_hyperparameters(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def set_random_seeds(seed=42):
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# ------------------------------------------------------------------ #
# k-fold training
# ------------------------------------------------------------------ #
def train_model(cfg_path="config.yaml"):
    hp = load_hyperparameters(cfg_path)
    k_folds = hp["K_FOLDS"]

    run_dir = os.path.join("modelos_kf", f"{hp['NAME_DATASET']}_vector")
    os.makedirs(run_dir, exist_ok=True)

    metrics_hist = {}

    for fold in range(k_folds):
        print(f"\n==================== Fold {fold+1}/{k_folds} ====================")

        ckpt_cb = ModelCheckpoint(
            dirpath   = run_dir,
            filename  = f"fold_{fold}_best_model",
            monitor   = "val_loss",
            mode      = "min",
            save_top_k= 1
        )

        early_cb = EarlyStoppingAtSpecificEpoch(
            patience   = 2,
            threshold  = 1e-3,
            monitor    = "val_loss",
            mode       = "min",
            verbose    = True
        )

        model = ReLuMLP2L(
            name_dataset    = hp["NAME_DATASET"],
            shape           = hp["SHAPE"],
            epochs          = hp["MAX_EPOCHS"],
            learning_rate   = hp["LEARNING_RATE"],
            features_dim    = hp["FEATURES_DIM"],
            drop_path_rate  = hp["DROP_PATH_RATE"],       # aceito mas ignorado
            num_classes     = hp["NUM_CLASSES"],
            label_smoothing = hp["LABEL_SMOOTHING"],
            optimizer_momentum=(hp["OPTIMIZER_MOMENTUM"], 0.999),
            weight_decay    = hp["WEIGHT_DECAY"],
            layer_scale     = hp["LAYER_SCALE"]
        )

        dm = CustomImageCSVModule_kf(
            train_dir   = hp["TRAIN_DIR"],
            test_dir    = hp["TEST_DIR"],
            shape       = hp["SHAPE"],
            batch_size  = hp["BATCH_SIZE"],
            num_workers = hp["NUM_WORKERS"],
            n_splits    = k_folds,
            fold_idx    = fold
        )
        dm.setup("fit")

        trainer = pl.Trainer(
            log_every_n_steps = 10,
            accelerator = hp["ACCELERATOR"],
            devices     = hp["DEVICES"],
            precision   = hp["PRECISION"],
            max_epochs  = hp["MAX_EPOCHS"],
            callbacks   = [TQDMProgressBar(leave=True), ckpt_cb, early_cb]
        )

        trainer.fit(model, dm)

        if ckpt_cb.best_model_path == "":
            print("⚠️  Early-stopped antes de salvar melhor modelo; pulando avaliação.")
            continue

        # ---------------- avaliação ----------------
        best_model = CustomVectorModel.load_from_checkpoint(ckpt_cb.best_model_path)
        val_metrics  = trainer.validate(best_model, dm, verbose=False)[0]
        test_metrics = trainer.test(best_model, dm, verbose=False)[0]

        for k, v in {**val_metrics, **test_metrics}.items():
            metrics_hist.setdefault(k, []).append(v)

        # cleanup
        del model, best_model
        torch.cuda.empty_cache()

    # ------------ resumo final ------------
    print("\n==================== Métricas Finais ====================")
    for m, vals in metrics_hist.items():
        if isinstance(vals[0], (int, float, np.floating)):
            print(f"{m}: mean = {np.mean(vals):.4f}, std = {np.std(vals):.4f}")

# ------------------------------------------------------------------ #
if __name__ == "__main__":
    set_random_seeds()
    train_model()
