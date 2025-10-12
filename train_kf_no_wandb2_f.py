import os
import shutil
import random
import yaml
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import time

# ---------------------------------------------------------
# Determinismo CUDA (tem que vir ANTES do import torch)
# ---------------------------------------------------------
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint, Callback

# Importa apenas a versão atual do modelo (sem ensemble)
from model import CustomEnsembleModel
from kf_data import CustomImageModule_kf
# (sem callbacks extras não utilizados)

# -----------------------------------------------------------------------------
# Utilidades auxiliares
# -----------------------------------------------------------------------------

def load_hyperparameters(file_path: str) -> Dict[str, Any]:
    with open(file_path, "r") as file:
        return yaml.safe_load(file)

def set_random_seeds(seed: int):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def time_block() -> Tuple[float, callable]:
    start = time.perf_counter()
    def stop():
        return time.perf_counter() - start
    return start, stop

def write_header_if_new(txt_path: str, hparams: Dict[str, Any], seed: int):
    if not os.path.exists(txt_path):
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"# Resultados — Seed {seed}\n")
            f.write(f"Data/Hora: {datetime.now().isoformat(timespec='seconds')}\n\n")
            f.write("## Hiperparâmetros\n")
            yaml.dump(hparams, f, allow_unicode=True, sort_keys=False)
            f.write("\n## Resultados por fold\n")

def append_fold_result(
    txt_path: str,
    fold: int,
    train_time_s: float,
    test_time_s: float,
    best_acc: float,
    best_ckpt: str,
    ckpt_candidates: List[str],
    metrics_csv_path: Optional[str] = None,
    last_train_metrics: Optional[Dict[str, float]] = None,
    last_val_metrics: Optional[Dict[str, float]] = None,
):
    with open(txt_path, "a", encoding="utf-8") as f:
        f.write(
            f"\n### Fold {fold}\n"
            f"- Tempo de treino (s): {train_time_s:.3f}\n"
            f"- Tempo de teste (s): {test_time_s:.3f}\n"
            f"- Melhor test_accuracy: {best_acc:.6f}\n"
            f"- Checkpoint escolhido: {best_ckpt}\n"
        )
        if metrics_csv_path:
            f.write(f"- Métricas por época (CSV): {metrics_csv_path}\n")
        if last_train_metrics:
            f.write("- Últimas métricas de treino registradas:\n")
            for k, v in sorted(last_train_metrics.items()):
                f.write(f"  - {k}: {v:.6f}\n")
        if last_val_metrics:
            f.write("- Últimas métricas de validação registradas:\n")
            for k, v in sorted(last_val_metrics.items()):
                f.write(f"  - {k}: {v:.6f}\n")
        f.write(f"- Avaliados ({len(ckpt_candidates)}):\n")
        for p in ckpt_candidates:
            f.write(f"  - {p}\n")

def write_seed_summary(txt_path: str, accs: List[float]):
    if not accs:
        return
    mean, std = float(np.mean(accs)), float(np.std(accs))
    with open(txt_path, "a", encoding="utf-8") as f:
        f.write(
            "\n## Resumo da seed\n"
            f"- test_accuracy mean: {mean:.6f}\n"
            f"- test_accuracy std:  {std:.6f}\n"
            f"- n (folds):          {len(accs)}\n"
        )

def write_global_summary(base_dir: str, metrics_history: Dict[str, List[float]]):
    resumo_path = os.path.join(base_dir, "resumo_resultados.txt")
    with open(resumo_path, "w", encoding="utf-8") as f:
        f.write(f"# Resumo Global — {datetime.now().isoformat(timespec='seconds')}\n\n")
        for name, values in metrics_history.items():
            if not values:
                continue
            mean, std = float(np.mean(values)), float(np.std(values))
            f.write(
                f"{name}: mean = {mean:.6f}, std = {std:.6f}  "
                f"(n={len(values)}; melhores por fold/seed)\n"
            )

# -----------------------------------------------------------------------------
# Callback: grava métricas por época em CSV
# -----------------------------------------------------------------------------

class EpochMetricsToFileCallback(Callback):
    """
    Grava métricas por época em CSV.
    - Detecta automaticamente as chaves de métricas de train/val em `trainer.callback_metrics`.
    - Escreve cabeçalho na primeira chamada de validação.
    - Por padrão filtra chaves que contenham 'train' ou 'val' no nome (case-insensitive).
    - Mantém também as últimas métricas vistas para sumarizar no txt.
    """
    def __init__(self, csv_path: str, include_patterns: Tuple[str, ...] = ("train", "val")):
        super().__init__()
        self.csv_path = csv_path
        self.include_patterns = tuple(s.lower() for s in include_patterns)
        self._header_written = False
        self._columns: List[str] = []
        self.last_train_metrics: Dict[str, float] = {}
        self.last_val_metrics: Dict[str, float] = {}

        # prepara diretório
        ensure_dir(os.path.dirname(csv_path))

    @staticmethod
    def _to_float(x) -> Optional[float]:
        try:
            if isinstance(x, (float, int)):
                return float(x)
            if hasattr(x, "item"):
                return float(x.item())
        except Exception:
            return None
        return None

    def _select_metric_keys(self, metrics: Dict[str, Any]) -> List[str]:
        keys = []
        for k, v in metrics.items():
            k_low = k.lower()
            if any(pat in k_low for pat in self.include_patterns):
                # evita chaves altamente voláteis/auxiliares
                if k.startswith("epoch") or k.endswith("_step") or k.endswith("_epoch"):
                    # epoch será tratado separadamente
                    pass
                keys.append(k)
        # ordenar para reprodutibilidade
        keys = sorted(set(keys))
        return keys

    def _write_header(self, trainer: pl.Trainer):
        metrics = dict(trainer.callback_metrics)
        # epoch corrente
        epoch = trainer.current_epoch if trainer is not None else 0
        # descobre colunas
        self._columns = self._select_metric_keys(metrics)
        # garante 'epoch' como primeira coluna
        header_cols = ["epoch"] + self._columns
        with open(self.csv_path, "w", encoding="utf-8") as f:
            f.write(",".join(header_cols) + "\n")
        self._header_written = True

    def _write_row(self, trainer: pl.Trainer):
        metrics = dict(trainer.callback_metrics)
        if not self._header_written:
            self._write_header(trainer)
        row = []
        epoch = int(getattr(trainer, "current_epoch", 0))
        row.append(str(epoch))
        # atualiza últimos dicionários amigáveis
        cur_train, cur_val = {}, {}
        for k in self._columns:
            val = self._to_float(metrics.get(k))
            row.append("" if val is None else f"{val:.8f}")
            lk = k.lower()
            if "train" in lk and val is not None:
                cur_train[k] = val
            if "val" in lk and val is not None:
                cur_val[k] = val

        # salva linha
        with open(self.csv_path, "a", encoding="utf-8") as f:
            f.write(",".join(row) + "\n")

        # guarda últimos valores conhecidos
        if cur_train:
            self.last_train_metrics = cur_train
        if cur_val:
            self.last_val_metrics = cur_val

    # Chamado ao fim de cada época de validação
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._write_row(trainer)

# -----------------------------------------------------------------------------
# Função principal de treino (k-fold)
# -----------------------------------------------------------------------------

def train_model(config_path: str = "config.yaml"):
    hparams = load_hyperparameters(config_path)
    k_splits: int = hparams["K_FOLDS"]

    # Diretório base do experimento (sem seed)
    base_dir = os.path.join("modelos_kf", f"{hparams['NAME_DATASET']}_{hparams['TMODEL']}_ne")
    ensure_dir(base_dir)

    metrics_history: Dict[str, List[float]] = {}

    # Seeds: 42 .. 42+N_SEEDS-1
    for seed in range(42, 42 + hparams["N_SEEDS"]):
        set_random_seeds(seed)

        run_dir = os.path.join(base_dir, f"seed_{seed:03d}")
        ensure_dir(run_dir)
        final_model_dir = os.path.join(run_dir, "final_best_models")
        ensure_dir(final_model_dir)

        seed_txt = os.path.join(run_dir, f"seed_{seed:03d}_resultados.txt")
        write_header_if_new(seed_txt, hparams, seed)

        seed_best_accs: List[float] = []

        for fold in range(k_splits):
            print(f"\n==================== Seed {seed} | Fold {fold + 1}/{k_splits} ====================")

            # Caminho do CSV de métricas por época
            metrics_csv_path = os.path.join(run_dir, f"fold_{fold}_epoch_metrics.csv")
            epoch_metrics_cb = EpochMetricsToFileCallback(metrics_csv_path)

            # Callbacks (mantém padrão de checkpoints na raiz da seed)
            ckpt_callback = ModelCheckpoint(
                dirpath=run_dir,
                filename=f"fold_{fold}_best_model_{{val_loss:.4f}}",
                monitor="val_loss",
                mode="min",
                save_top_k=3,
            )
            callbacks = [TQDMProgressBar(leave=True), ckpt_callback, epoch_metrics_cb]

            # Dados
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

            # Modelo
            model = CustomEnsembleModel(
                tmodel=hparams["TMODEL"],
                name_dataset=hparams["NAME_DATASET"],
                epochs=hparams["MAX_EPOCHS"],
                shape=hparams["SHAPE"],
                learning_rate=hparams["LEARNING_RATE"],
                drop_path_rate=hparams["DROP_PATH_RATE"],
                num_classes=hparams["NUM_CLASSES"],
                label_smoothing=hparams["LABEL_SMOOTHING"],
                optimizer_momentum=(hparams["OPTIMIZER_MOMENTUM"], 0.999),
                weight_decay=hparams("WEIGHT_DECAY"),
                layer_scale=hparams["LAYER_SCALE"]
            )

            # Trainer
            trainer = pl.Trainer(
                log_every_n_steps=10,
                accelerator=hparams["ACCELERATOR"],
                devices=hparams["DEVICES"],
                precision=hparams["PRECISION"],
                max_epochs=hparams["MAX_EPOCHS"],
                callbacks=callbacks,
                deterministic=True,
            )

            # Treino (tempo)
            _, stop_train = time_block()
            trainer.fit(model, data_module)
            train_time_s = stop_train()

            # Avalia top-3 checkpoints do fold
            checkpoint_files = sorted(
                [
                    os.path.join(run_dir, fname)
                    for fname in os.listdir(run_dir)
                    if fname.startswith(f"fold_{fold}_best_model_") and fname.endswith(".ckpt")
                ]
            )

            try:
                data_module.setup(stage="test")
            except Exception:
                pass

            best_accuracy = -1.0
            best_model_path = None
            test_time_total = 0.0

            for ckpt_path in checkpoint_files:
                eval_model = CustomModel.load_from_checkpoint(ckpt_path)
                _, stop_test = time_block()
                test_metrics = trainer.test(eval_model, data_module, verbose=False)[0]
                test_time = stop_test()
                test_time_total += test_time

                test_accuracy = float(test_metrics.get("test_accuracy", 0.0))
                if test_accuracy > best_accuracy:
                    best_accuracy = test_accuracy
                    best_model_path = ckpt_path

            # Copia melhor e registra no TXT da seed
            if best_model_path:
                dest = os.path.join(final_model_dir, f"fold_{fold}_best_model.ckpt")
                shutil.copy(best_model_path, dest)
                print(
                    f"Melhor modelo do Fold {fold} (seed={seed}) salvo em '{dest}' "
                    f"(test_accuracy={best_accuracy:.6f})"
                )
                metrics_history.setdefault("test_accuracy", []).append(best_accuracy)
                seed_best_accs.append(best_accuracy)

                # Coleta últimas métricas vistas pelo callback (para sumarizar no txt)
                last_train = epoch_metrics_cb.last_train_metrics
                last_val = epoch_metrics_cb.last_val_metrics

                append_fold_result(
                    seed_txt,
                    fold=fold,
                    train_time_s=train_time_s,
                    test_time_s=test_time_total,
                    best_acc=best_accuracy,
                    best_ckpt=dest,
                    ckpt_candidates=checkpoint_files,
                    metrics_csv_path=metrics_csv_path,
                    last_train_metrics=last_train if last_train else None,
                    last_val_metrics=last_val if last_val else None,
                )
            else:
                append_fold_result(
                    seed_txt,
                    fold=fold,
                    train_time_s=train_time_s,
                    test_time_s=test_time_total,
                    best_acc=-1.0,
                    best_ckpt="N/A",
                    ckpt_candidates=checkpoint_files,
                    metrics_csv_path=metrics_csv_path,
                    last_train_metrics=epoch_metrics_cb.last_train_metrics or None,
                    last_val_metrics=epoch_metrics_cb.last_val_metrics or None,
                )

        write_seed_summary(seed_txt, seed_best_accs)

    # ---- Métricas finais (console + arquivo agregado) ----
    print("\n==================== Métricas Finais ====================")
    for name, values in metrics_history.items():
        mean, std = np.mean(values), np.std(values)
        print(f"{name}: mean = {mean:.6f}, std = {std:.6f}  "
              f"(n={len(values)}; melhores por fold/seed)")
    write_global_summary(base_dir, metrics_history)

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    train_model()
