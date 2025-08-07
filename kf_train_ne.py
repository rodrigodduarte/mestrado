#!/usr/bin/env python3
"""
kf_train_triple_sweep.py
────────────────────────
• Treina **somente o primeiro fold** (fold 0) do modelo TRIPLO
  (ConvNeXt + Swin + vetor) usando busca *sweep* do W&B.
• A estrutura, callbacks e hiper-parâmetros seguem o
  `train_kf_no_wandb.py` original, mas:
    – substitui `CustomEnsembleModel` → `CustomModelTriple`;
    – usa sempre o `CustomImageCSVModule_kf` (images + feature vector);
    – salva em   modelos_kf/<DATASET>_triple/.

Execute:

    conda activate mestrado
    python kf_train_triple_sweep.py
"""

# ───────────── imports básicos ───────────── #
import os, random, yaml, numpy as np, torch, pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
import wandb

# ───────────── imports do projeto ────────── #
from model import CustomModel
from kf_data import CustomImageCSVModule_kf
# callbacks auxiliares (se quiser manter) —
from callbacks import EarlyStoppingAtSpecificEpoch, SaveBestOrLastModelCallback

# ╭──────────────── utilidades ───────────────╮
def load_hyperparameters(path="config.yaml"):
    with open(path, "r") as f: return yaml.safe_load(f)

def set_random_seeds(seed=42):
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
# ╰────────────────────────────────────────────╯


def train_model(config=None):
    hp = load_hyperparameters()
    run_dir = os.path.join("modelos_kf", f"{hp['NAME_DATASET']}_triple")
    os.makedirs(run_dir, exist_ok=True)

    with wandb.init(project=hp["PROJECT"], config=config):
        cfg = wandb.config
        wandb_logger = WandbLogger(project=hp["PROJECT"])

        # ───────────────── fold 0 apenas ────────────────
        fold = 0
        ckpt_cb = ModelCheckpoint(
            dirpath=run_dir,
            filename=f"fold_{fold}_best_model",
            monitor="val_loss",
            mode="min",
            save_top_k=1
        )

        model = CustomModel(
            name_dataset   = hp["NAME_DATASET"],
            shape          = hp["SHAPE"],
            epochs         = hp["MAX_EPOCHS"],
            learning_rate  = float(cfg.learning_rate),
            features_dim   = hp["FEATURES_DIM"],
            drop_path_rate = cfg.drop_path_rate,
            num_classes    = hp["NUM_CLASSES"],
            label_smoothing= cfg.label_smoothing,
            optimizer_momentum=(cfg.optimizer_momentum, 0.999),
            weight_decay   = float(cfg.weight_decay),
            layer_scale    = cfg.layer_scale
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
            callbacks     = [TQDMProgressBar(leave=True), ckpt_cb]
        )

        trainer.fit(model, dm)

        # ─────────── avaliação do melhor checkpoint ───────────
        best_model = CustomModel.load_from_checkpoint(ckpt_cb.best_model_path)
        val_metrics  = trainer.validate(best_model, dm)[0]
        test_metrics = trainer.test(best_model, dm)[0]

        wandb.log({f"fold0_{k}": v for k, v in {**val_metrics, **test_metrics}.items()})


# ╭───────────────── entrada CLI ─────────────────╮
if __name__ == "__main__":
    set_random_seeds()

    sweep_conf = {
        'method': 'random',
        'metric': {'name': 'val_loss', 'goal': 'minimize'},
        'parameters': {
            'learning_rate'          : {'min': 3e-5,  'max': 9e-5, 'distribution': 'log_uniform_values'},
            'weight_decay'           : {'min': 3e-5,  'max': 4e-4, 'distribution': 'log_uniform_values'},
            'optimizer_momentum'     : {'min': 0.95,  'max': 0.99, 'distribution': 'uniform'},
            'layer_scale'            : {'min': 0.8,   'max': 1.5,  'distribution': 'uniform'},
            'drop_path_rate'         : {'min': 0.02,   'max': 0.1,  'distribution': 'uniform'},
            'label_smoothing'        : {'min': 0.0,   'max': 0.1,  'distribution': 'uniform'},
        }
    }

    project_name = load_hyperparameters()["PROJECT"]
    sweep_id = wandb.sweep(sweep_conf, project=project_name)
    wandb.agent(sweep_id, function=train_model, count=200)
    wandb.finish()
