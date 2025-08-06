#!/usr/bin/env python3
"""
kf_train_vector_sweep.py
────────────────────────
Busca hiper-parâmetros (W&B sweep) **apenas para o modelo baseado no vetor de
características** — mesma assinatura do modelo triplo, mas sem usar imagens.

Diferenças em relação ao kf_train_triple.py
• Classe de modelo  →  CustomVectorModel
• Pasta de saída    →  modelos_kf/<DATASET>_vector/
• DataModule        →  continua CustomImageCSVModule_kf
  (ele já entrega (img, features, label); o modelo ignora a imagem)

Execute:

    conda activate mestrado
    python kf_train_vector_sweep.py
"""

# ───────────── imports básicos ───────────── #
import os, random, yaml, numpy as np, torch, pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
import wandb

# ───────────── imports do projeto ────────── #
from model import CustomVectorModel          # ← nova classe “only-vector”
from kf_data import CustomImageCSVModule_kf
from callbacks import EarlyStoppingAtSpecificEpoch

# ╭──────────────── utilidades ───────────────╮
def hyperparams(path="config.yaml"):
    with open(path, "r") as f: return yaml.safe_load(f)

def fix_seeds(seed=42):
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
# ╰────────────────────────────────────────────╯


def train_model(cfg_sweep=None):
    hp = hyperparams()
    run_dir = os.path.join("modelos_kf", f"{hp['NAME_DATASET']}_vector")
    os.makedirs(run_dir, exist_ok=True)

    with wandb.init(project=hp["PROJECT"], config=cfg_sweep):
        cfg = wandb.config
        wandb_logger = WandbLogger(project=hp["PROJECT"])

        fold = 0
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

        model = CustomVectorModel(
            name_dataset    = hp["NAME_DATASET"],
            shape           = hp["SHAPE"],
            epochs          = hp["MAX_EPOCHS"],
            learning_rate   = float(cfg.learning_rate),
            features_dim    = hp["FEATURES_DIM"],
            drop_path_rate  = cfg.drop_path_rate,     # aceito mas não usado
            num_classes     = hp["NUM_CLASSES"],
            label_smoothing = cfg.label_smoothing,
            optimizer_momentum=(cfg.optimizer_momentum, 0.999),
            weight_decay    = float(cfg.weight_decay),
            layer_scale     = cfg.layer_scale
        )

        dm = CustomImageCSVModule_kf(
            train_dir   = hp["TRAIN_DIR"],
            test_dir    = hp["TEST_DIR"],
            shape       = hp["SHAPE"],
            batch_size  = hp["BATCH_SIZE"],
            num_workers = hp["NUM_WORKERS"],
            n_splits    = hp["K_FOLDS"],
            fold_idx    = fold
        )
        dm.setup(stage="fit")

        trainer = pl.Trainer(
            logger        = wandb_logger,
            log_every_n_steps = 10,
            accelerator   = hp["ACCELERATOR"],
            devices       = hp["DEVICES"],
            precision     = hp["PRECISION"],
            max_epochs    = hp["MAX_EPOCHS"],
            callbacks     = [TQDMProgressBar(leave=True), ckpt_cb, early_cb]
        )

        trainer.fit(model, dm)

        # ——— pula avaliação se early stop cancelou antes de salvar ———
        if ckpt_cb.best_model_path == "":
            wandb.log({"status": "early_stopped"})
            return

        best_model  = CustomVectorModel.load_from_checkpoint(ckpt_cb.best_model_path)
        val_metrics = trainer.validate(best_model, dm, verbose=False)[0]
        test_metrics= trainer.test(best_model, dm, verbose=False)[0]

        wandb.log({f"fold0_{k}": v for k, v in {**val_metrics, **test_metrics}.items()})


# ╭────────────────── Sweep config ──────────────────╮
if __name__ == "__main__":
    fix_seeds()

    sweep_conf = {
        "method": "random",
        "metric": {"name": "val_loss", "goal": "minimize"},
        "parameters": {
            "learning_rate"      : {"min": 3e-5,  "max": 9e-5,  "distribution": "log_uniform_values"},
            "weight_decay"       : {"min": 3e-5,  "max": 4e-4,  "distribution": "log_uniform_values"},
            "optimizer_momentum" : {"min": 0.95,  "max": 0.99,  "distribution": "uniform"},
            "layer_scale"        : {"min": 0.8,   "max": 1.5,   "distribution": "uniform"},
            "drop_path_rate"     : {"min": 0.0,   "max": 0.1,   "distribution": "uniform"},
            "label_smoothing"    : {"min": 0.0,   "max": 0.1,   "distribution": "uniform"},
        }
    }

    sweep_id = wandb.sweep(sweep_conf, project=hyperparams()["PROJECT"])
    wandb.agent(sweep_id, function=train_model, count=200)
    wandb.finish()
