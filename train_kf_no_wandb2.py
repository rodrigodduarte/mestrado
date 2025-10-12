import os
import shutil
import random
import yaml
from typing import Dict, Any, List

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint

from model import CustomModel
from kf_data import CustomImageModule_kf
# Removi callbacks não usados para limpar; se quiser usar, re-add e inclua no `callbacks`.
# from callbacks import (EarlyStoppingAtSpecificEpoch, SaveBestOrLastModelCallback, EarlyStopCallback)

# -----------------------------------------------------------------------------
# Utilidades auxiliares
# -----------------------------------------------------------------------------

def load_hyperparameters(file_path: str) -> Dict[str, Any]:
    with open(file_path, "r") as file:
        return yaml.safe_load(file)

def set_random_seeds(seed: int):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Se quiser ser estrito: torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# -----------------------------------------------------------------------------
# Função principal de treino (k-fold)
# -----------------------------------------------------------------------------

def train_model(config_path: str = "config2.yaml"):
    hparams = load_hyperparameters(config_path)
    k_splits: int = hparams["K_FOLDS"]

    # Diretório base do experimento (sem seed)
    base_dir = os.path.join("modelos_kf", f"{hparams['NAME_DATASET']}_{hparams['TMODEL']}_ne")
    os.makedirs(base_dir, exist_ok=True)

    metrics_history: Dict[str, List[float]] = {}

    # Agora garante exatamente N_SEEDS seeds: 42 .. 42+N_SEEDS-1
    for seed in range(42, 42 + hparams["N_SEEDS"]):
        set_random_seeds(seed)

        # Subpasta por seed para não colidir checkpoints
        run_dir = os.path.join(base_dir, f"seed_{seed:03d}")
        os.makedirs(run_dir, exist_ok=True)
        final_model_dir = os.path.join(run_dir, "final_best_models")
        os.makedirs(final_model_dir, exist_ok=True)

        for fold in range(k_splits):
            print(f"\n==================== Seed {seed} | Fold {fold + 1}/{k_splits} ====================")

            # ----- Callbacks -----
            ckpt_callback = ModelCheckpoint(
                dirpath=run_dir,
                filename=f"fold_{fold}_seed{seed:03d}_best_{{val_loss:.4f}}",
                monitor="val_loss",
                mode="min",
                save_top_k=3,
            )
            callbacks = [TQDMProgressBar(leave=True), ckpt_callback]

            # ----- Dados -----
            data_module = CustomImageModule_kf(
                train_dir=hparams["TRAIN_DIR"],
                test_dir=hparams["TEST_DIR"],
                shape=hparams["SHAPE"],
                batch_size=hparams["BATCH_SIZE"],
                num_workers=hparams["NUM_WORKERS"],
                n_splits=k_splits,
                fold_idx=fold,
            )
            data_module.setup(stage="fit")

            # ----- Modelo -----
            model = CustomModel(
                tmodel=hparams["TMODEL"],
                name_dataset=hparams["NAME_DATASET"],
                epochs=hparams["MAX_EPOCHS"],
                shape=hparams["SHAPE"],
                learning_rate=hparams["LEARNING_RATE"],
                drop_path_rate=hparams["DROP_PATH_RATE"],
                num_classes=hparams["NUM_CLASSES"],
                label_smoothing=hparams["LABEL_SMOOTHING"],
                optimizer_momentum=(hparams["OPTIMIZER_MOMENTUM"], 0.999),
                weight_decay=hparams.get("WEIGHT_DECAY", 0.0),
            )

            # ----- Trainer -----
            trainer = pl.Trainer(
                log_every_n_steps=10,
                accelerator=hparams["ACCELERATOR"],
                devices=hparams["DEVICES"],
                precision=hparams["PRECISION"],
                max_epochs=hparams["MAX_EPOCHS"],
                callbacks=callbacks,
            )

            # ----- Treino -----
            trainer.fit(model, data_module)

            # ----- Avaliação dos top-3 checkpoints do fold/seed -----
            checkpoint_files = sorted(
                [
                    os.path.join(run_dir, fname)
                    for fname in os.listdir(run_dir)
                    if fname.startswith(f"fold_{fold}_seed{seed:03d}_best_") and fname.endswith(".ckpt")
                ]
            )

            # Garantir que data_module está pronto para teste (se necessário)
            try:
                data_module.setup(stage="test")
            except Exception:
                pass  # caso o setup('test') não seja implementado

            best_accuracy = -1.0
            best_model_path = None

            for ckpt_path in checkpoint_files:
                eval_model = CustomModel.load_from_checkpoint(ckpt_path)
                test_metrics = trainer.test(eval_model, data_module, verbose=False)[0]
                test_accuracy = float(test_metrics.get("test_accuracy", 0.0))
                if test_accuracy > best_accuracy:
                    best_accuracy = test_accuracy
                    best_model_path = ckpt_path

            if best_model_path:
                dest = os.path.join(final_model_dir, f"fold_{fold}_seed{seed:03d}_best_model.ckpt")
                shutil.copy(best_model_path, dest)
                print(
                    f"Melhor modelo do Fold {fold} (seed={seed}) salvo em '{dest}' "
                    f"(test_accuracy={best_accuracy:.4f})"
                )
                # Agora guardamos APENAS o melhor por fold (correto para estatística)
                metrics_history.setdefault("test_accuracy", []).append(best_accuracy)

    # ------------------------ MÉTRICAS FINAIS -----------------------------
    print("\n==================== Métricas Finais ====================")
    for name, values in metrics_history.items():
        mean, std = np.mean(values), np.std(values)
        print(f"{name}: mean = {mean:.4f}, std = {std:.4f}  "
              f"(n={len(values)}; agregado de melhores por fold/seed)")
    
if __name__ == "__main__":
    train_model()
