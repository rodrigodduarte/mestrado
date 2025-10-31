#!/usr/bin/env python3
# extracao_vector_only.py
# -----------------------------------------------------------------------------
# Benchmark/estatísticas APENAS para o modelo baseado em vetor ("vector"/SSN).
# Mantém o estilo do arquivo anexado (extracao.py), alterando só o necessário:
#   - Executa somente TMODEL == "vector";
#   - Aceita config1.yaml OU config.yaml (primeiro que existir);
#   - Usa CustomFeaturesFromFoldersModule_kf e CustomFeaturesOnlyModel;
#   - Infere automaticamente FEATURES_DIM a partir do DataModule se não vier no YAML;
#   - Não salva checkpoints; apenas medidas de treino/teste e métricas.
# Saída: treinamentos/estatisticas/<dataset>_vector/fold_k/{json,txt} + resumo geral.
# -----------------------------------------------------------------------------

import os
import time
import json
import yaml
import random
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar

# Importa implementações locais do usuário (iguais ao extracao.py original)
from model import (
    CustomFeaturesOnlyModel,
)

from dataset import (
    # DataModule apenas de vetores de atributos
    CustomFeaturesFromFoldersModule_kf,
)

# -------------------------------------------------
# utilidades básicas
# -------------------------------------------------

def load_hparams(path_yaml: str):
    with open(path_yaml, "r") as f:
        return yaml.safe_load(f)


def set_seeds(seed: int):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def len_dataloader_safe(dl):
    """Conta amostras do dataloader de maneira robusta."""
    try:
        return len(dl.dataset)
    except Exception:
        n = 0
        for b in dl:
            x = b[0] if isinstance(b, (list, tuple)) and len(b) >= 1 else b
            try:
                n += x.size(0)
            except Exception:
                try:
                    n += len(x)
                except Exception:
                    n += 1
        return n


def infer_features_dim_from_features_dm(dm):
    """Infere a dimensionalidade do vetor de atributos em um DataModule só-vetor.
    Assumimos batch no formato (features, labels)."""
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    x = batch[0] if isinstance(batch, (list, tuple)) else batch
    if isinstance(x, (list, tuple)):
        x = x[0]
    x = torch.as_tensor(x)
    if x.ndim == 1:
        return int(x.shape[0])
    return int(x.shape[-1])


# -------------------------------------------------
# construção do DataModule e do Modelo (apenas vector)
# -------------------------------------------------

def build_datamodule(h, n_splits: int, fold_idx: int):
    assert h["TMODEL"] == "vector", "Este script suporta apenas TMODEL == 'vector'."
    dm = CustomFeaturesFromFoldersModule_kf(
        train_dir=h["TRAIN_DIR"],
        test_dir=h["TEST_DIR"],
        shape=h["SHAPE"],
        batch_size=h["BATCH_SIZE"],
        num_workers=h["NUM_WORKERS"],
        n_splits=n_splits,
        fold_idx=fold_idx,
    )
    return dm


def build_model(h, dm=None):
    assert h["TMODEL"] == "vector", "Este script suporta apenas TMODEL == 'vector'."

    opt_mom = h["OPTIMIZER_MOMENTUM"]
    if isinstance(opt_mom, (int, float)):
        opt_mom = (opt_mom, 0.999)
    elif isinstance(opt_mom, list):
        opt_mom = tuple(opt_mom)

    features_dim = h.get("FEATURES_DIM", None)
    if features_dim is None:
        assert dm is not None, "Preciso do DataModule para inferir FEATURES_DIM."
        features_dim = infer_features_dim_from_features_dm(dm)

    model = CustomFeaturesOnlyModel(
        name_dataset=h["NAME_DATASET"],
        shape=h["SHAPE"],
        epochs=h["MAX_EPOCHS"],
        learning_rate=h["LEARNING_RATE"],
        features_dim=features_dim,
        drop_path_rate=h["DROP_PATH_RATE"],
        num_classes=h["NUM_CLASSES"],
        label_smoothing=h["LABEL_SMOOTHING"],
        optimizer_momentum=opt_mom,
        weight_decay=h["WEIGHT_DECAY"],
        layer_scale=h["LAYER_SCALE"],
    )
    return model


# -------------------------------------------------
# estatística e agregação entre folds
# -------------------------------------------------

def mean_std(values):
    vals = [v for v in values if v is not None]
    if len(vals) == 0:
        return (None, None)
    if len(vals) == 1:
        return (float(vals[0]), 0.0)
    arr = np.array(vals, dtype=float)
    return (float(np.mean(arr)), float(np.std(arr, ddof=1)))


def aggregate_reports(reports, dataset_name):
    train_times, test_times = [], []
    inf_ms, throughput = [], []
    gpu_train, gpu_test = [], []
    model_mb, nparams = [], []

    acc_list, err_list = [], []
    f1_list, prec_list, rec_list, loss_list = [], [], [], []

    for r in reports:
        train_times.append(r["train_time_sec"])
        test_times.append(r["test_time_sec"])
        inf_ms.append(r["test_inf_ms_per_sample"])
        throughput.append(r["throughput_samples_per_sec"])
        gpu_train.append(r["max_gpu_mem_mb_train"])
        gpu_test.append(r["max_gpu_mem_mb_test"])
        model_mb.append(r["model_size_mb"])
        nparams.append(r["num_params_trainable"])

        tm = r["test_metrics"]
        acc = tm.get("test_accuracy")
        f1  = tm.get("test_f1")
        prc = tm.get("test_precision")
        rc  = tm.get("test_recall")
        los = tm.get("test_loss")
        if acc is not None:
            acc_list.append(acc)
            err_list.append(1.0 - acc)
        if f1 is not None:
            f1_list.append(f1)
        if prc is not None:
            prec_list.append(prc)
        if rc is not None:
            rec_list.append(rc)
        if los is not None:
            loss_list.append(los)

    return {
        "dataset": dataset_name,
        "tmodel": "vector",
        "num_folds_ok": len(reports),
        # custo computacional
        "train_time_sec_mean_std": mean_std(train_times),
        "test_time_sec_mean_std": mean_std(test_times),
        "inf_ms_per_sample_mean_std": mean_std(inf_ms),
        "throughput_samples_per_sec_mean_std": mean_std(throughput),
        "max_gpu_mem_mb_train_mean_std": mean_std(gpu_train),
        "max_gpu_mem_mb_test_mean_std": mean_std(gpu_test),
        "model_size_mb_mean_std": mean_std(model_mb),
        "num_params_trainable_mean_std": mean_std(nparams),
        # desempenho
        "test_accuracy_mean_std": mean_std(acc_list),
        "test_error_mean_std": mean_std(err_list),
        "test_f1_mean_std": mean_std(f1_list),
        "test_precision_mean_std": mean_std(prec_list),
        "test_recall_mean_std": mean_std(rec_list),
        "test_loss_mean_std": mean_std(loss_list),
    }


def write_model_resumo(run_dir, agg):
    os.makedirs(run_dir, exist_ok=True)
    resumo_path = os.path.join(run_dir, "resumo.txt")

    def fmt(pair):
        if pair is None or pair[0] is None:
            return "N/A"
        m, s = pair
        return f"{m:.6f} ± {s:.6f}"

    with open(resumo_path, "w") as f:
        f.write(f"=== RESUMO {agg['dataset']} / {agg['tmodel']} ===\n")
        f.write(f"Folds válidos: {agg['num_folds_ok']}\n\n")
        f.write("== Desempenho de teste ==\n")
        f.write(f"Acurácia (mean ± std): {fmt(agg['test_accuracy_mean_std'])}\n")
        f.write(f"Test error (1-acc) (mean ± std): {fmt(agg['test_error_mean_std'])}\n")
        f.write(f"F1 macro (mean ± std): {fmt(agg['test_f1_mean_std'])}\n")
        f.write(f"Precision macro (mean ± std): {fmt(agg['test_precision_mean_std'])}\n")
        f.write(f"Recall macro (mean ± std): {fmt(agg['test_recall_mean_std'])}\n")
        f.write(f"Loss (mean ± std): {fmt(agg['test_loss_mean_std'])}\n\n")
        f.write("== Custo computacional ==\n")
        f.write(f"Tempo treino s (mean ± std): {fmt(agg['train_time_sec_mean_std'])}\n")
        f.write(f"Tempo teste s (mean ± std): {fmt(agg['test_time_sec_mean_std'])}\n")
        f.write(f"Latência ms/img (mean ± std): {fmt(agg['inf_ms_per_sample_mean_std'])}\n")
        f.write(f"Throughput img/s (mean ± std): {fmt(agg['throughput_samples_per_sec_mean_std'])}\n")
        f.write(f"GPU train MB pico (mean ± std): {fmt(agg['max_gpu_mem_mb_train_mean_std'])}\n")
        f.write(f"GPU test MB pico (mean ± std): {fmt(agg['max_gpu_mem_mb_test_mean_std'])}\n")
        f.write(f"Tamanho do modelo MB (mean ± std): {fmt(agg['model_size_mb_mean_std'])}\n")
        f.write(f"#params treináveis (mean ± std): {fmt(agg['num_params_trainable_mean_std'])}\n")


# -------------------------------------------------
# núcleo do benchmark de UM fold (1 época, sem ckpt)
# -------------------------------------------------

def _estimate_model_size_mb(model: torch.nn.Module):
    total_bytes = 0
    for p in model.parameters():
        total_bytes += p.numel() * p.element_size()
    return total_bytes / (1024 ** 2)


def benchmark_single_fold(h, fold_idx, base_save_dir, seed=42):
    set_seeds(seed)

    dataset_name = h["NAME_DATASET"]
    tmodel_name = h["TMODEL"]
    assert tmodel_name == "vector"

    run_name = f"{dataset_name}_{tmodel_name}"
    run_dir = os.path.join(base_save_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    fold_dir = os.path.join(run_dir, f"fold_{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)

    # salva hiperparâmetros desse fold
    with open(os.path.join(fold_dir, "hyperparams_used.json"), "w") as f:
        json.dump(h, f, indent=2, ensure_ascii=False)

    # DataModule do fold
    k_folds = int(h.get("K_FOLDS", 5))
    dm = build_datamodule(h, n_splits=k_folds, fold_idx=fold_idx)
    dm.setup(stage="fit")

    # Modelo
    model = build_model(h, dm=dm)

    # Trainer sem ModelCheckpoint
    trainer = pl.Trainer(
        logger=False,
        log_every_n_steps=10,
        accelerator=h["ACCELERATOR"],
        devices=h["DEVICES"],
        precision=h["PRECISION"],
        max_epochs=1,
        callbacks=[TQDMProgressBar(leave=True)],
    )

    # --- TREINO ---
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    trainer.fit(model, dm)
    train_time_sec = time.perf_counter() - t0
    max_gpu_mb_train = None
    if torch.cuda.is_available():
        max_gpu_mb_train = torch.cuda.max_memory_allocated() / (1024**2)

    epoch_trained = 0  # treinamos 1 época (epoch index 0)

    # --- VALIDAÇÃO ---
    val_metrics_list = trainer.validate(model, datamodule=dm, verbose=False)
    val_metrics = val_metrics_list[0] if len(val_metrics_list) > 0 else {}

    # --- TESTE ---
    try:
        dm.setup(stage="test")
    except Exception:
        pass

    test_dl = dm.test_dataloader()
    n_test = len_dataloader_safe(test_dl)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t1 = time.perf_counter()
    test_metrics_list = trainer.test(model, datamodule=dm, verbose=False)
    test_time_sec = time.perf_counter() - t1
    test_metrics = test_metrics_list[0] if len(test_metrics_list) > 0 else {}

    max_gpu_mb_test = None
    if torch.cuda.is_available():
        max_gpu_mb_test = torch.cuda.max_memory_allocated() / (1024**2)

    # latência média e throughput
    if n_test > 0 and test_time_sec > 0:
        inf_ms_per_sample = (test_time_sec * 1000.0) / n_test
        throughput_samples_per_sec = n_test / test_time_sec
    else:
        inf_ms_per_sample = float("nan")
        throughput_samples_per_sec = float("nan")

    # tamanho aproximado do modelo e #params
    model_size_mb = _estimate_model_size_mb(model)
    num_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    report = {
        "dataset": dataset_name,
        "tmodel": tmodel_name,
        "fold_idx": fold_idx,
        "epoch_trained": epoch_trained,
        "train_time_sec": train_time_sec,
        "test_time_sec": test_time_sec,
        "test_inf_ms_per_sample": inf_ms_per_sample,
        "throughput_samples_per_sec": throughput_samples_per_sec,
        "max_gpu_mem_mb_train": max_gpu_mb_train,
        "max_gpu_mem_mb_test": max_gpu_mb_test,
        "model_size_mb": model_size_mb,
        "num_params_trainable": num_params_trainable,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    with open(os.path.join(fold_dir, "benchmark_report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    with open(os.path.join(fold_dir, "benchmark_result.txt"), "w") as f:
        f.write("===== BENCHMARK REPORT (fold individual) =====\n")
        for k, v in report.items():
            f.write(f"{k}: {v}\n")

    print(f"[OK] {dataset_name}/vector/fold_{fold_idx} concluído (sem salvar modelo).")
    return report


# -------------------------------------------------
# main: roda somente TMODEL=vector, 1 seed fixa
# -------------------------------------------------

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_SAVE_DIR = os.path.join(BASE_DIR, "estatisticas")
    os.makedirs(BASE_SAVE_DIR, exist_ok=True)

    # Aceita config1.yaml OU config.yaml
    config_path = None
    for cand in ("config1.yaml", "config.yaml"):
        p = os.path.join(BASE_DIR, cand)
        if os.path.isfile(p):
            config_path = p
            break
    if config_path is None:
        raise FileNotFoundError("Nenhum config1.yaml ou config.yaml encontrado no diretório do script.")

    h_base = load_hparams(config_path)
    h_base["TMODEL"] = "vector"  # força o modo vetor

    dataset_name = h_base["NAME_DATASET"]
    k_folds = int(h_base.get("K_FOLDS", 5))
    SEED = 42  # seed fixa para benchmark reprodutível

    model_reports = []
    for fold_idx in range(k_folds):
        try:
            rep = benchmark_single_fold(
                h=h_base,
                fold_idx=fold_idx,
                base_save_dir=BASE_SAVE_DIR,
                seed=SEED,
            )
            model_reports.append(rep)
        except Exception as e:
            print(f"[WARN] Falhou {dataset_name}/vector/fold_{fold_idx}: {e}")

    if not model_reports:
        print("[SKIP] Nenhum fold executado para TMODEL=vector.")
        return

    agg = aggregate_reports(model_reports, dataset_name)

    run_name = f"{dataset_name}_vector"
    run_dir = os.path.join(BASE_SAVE_DIR, run_name)
    write_model_resumo(run_dir, agg)

    print("\n=== FINALIZADO. Resultados em treinamentos/estatisticas/ ===")


if __name__ == "__main__":
    main()
