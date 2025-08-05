#!/usr/bin/env python3
"""
estatisticas.py – Avalia os melhores modelos de cada fold, calcula métricas,
intervalos de confiança de 95 % (t-Student) e estatísticas-t, depois grava tudo
num arquivo .txt (com carimbo de data-hora) e exibe no console.
"""

# -------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------
import os
import random
from datetime import datetime
from pathlib import Path
import shutil  # se precisar mover algo depois
import yaml

import numpy as np
import torch
import pytorch_lightning as pl
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats  # IC95 + estatística-t

from model import CustomEnsembleModel
from kf_data import CustomImageCSVModule_kf

# -------------------------------------------------------------------------
# Utilidades
# -------------------------------------------------------------------------
def load_hyperparameters(cfg_path: str = "config2.yaml"):
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def set_random_seeds(seed: int = 42):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def plot_confusion_matrix(cm, save_path: str, title: str = "Matriz de Confusão"):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def print_final_stats(metric_list, name):
    arr = np.array(metric_list)
    print(f"{name} por Fold: {arr}")
    print(f"{name} Média: {arr.mean():.4f} | Desvio Padrão: {arr.std(ddof=1):.4f}\n")
    return arr.mean(), arr.std(ddof=1)


def ci95(mean, std, k):
    """Retorna (low, high) do IC95% via t-Student."""
    return stats.t.interval(
        0.95, df=max(k - 1, 1), loc=mean, scale=std / np.sqrt(k)
    )


def t_stat(mean, std, k, mu0=0.0):
    """t = (mean - mu0) / (std / sqrt(k))  (teste unilateral H0: μ = mu0)."""
    se = std / np.sqrt(k)
    return np.nan if se == 0 else (mean - mu0) / se


# -------------------------------------------------------------------------
# Configuração inicial
# -------------------------------------------------------------------------
set_random_seeds()
hyper = load_hyperparameters()  # assume config2.yaml na raiz

# Caminho base para salvar resultados do k-fold
base_dir = Path("modelos_kf") / f"{hyper['NAME_DATASET']}_{hyper['TMODEL']}"
base_dir.mkdir(parents=True, exist_ok=True)

# Listas para métricas por fold
acc_list, prec_list, rec_list, f1_list, loss_list = ([] for _ in range(5))
fold_metrics = {}

# -------------------------------------------------------------------------
# Loop de avaliação por fold
# -------------------------------------------------------------------------
for fold in range(hyper["K_FOLDS"]):
    ckpt_path = base_dir / f"fold_{fold}_best_model.ckpt"
    if not ckpt_path.exists():
        print(f"[Fold {fold}] – checkpoint não encontrado: {ckpt_path}")
        continue

    print(f"[Fold {fold}] Avaliando modelo: {ckpt_path}")
    model = CustomEnsembleModel.load_from_checkpoint(str(ckpt_path))
    model.eval()

    dm = CustomImageCSVModule_kf(
        train_dir=hyper["TRAIN_DIR"],
        test_dir=hyper["TEST_DIR"],
        shape=hyper["SHAPE"],
        batch_size=hyper["BATCH_SIZE"],
        num_workers=hyper["NUM_WORKERS"],
        n_splits=hyper["K_FOLDS"],
        fold_idx=fold,
    )
    dm.setup(stage="test")
    test_loader = dm.test_dataloader()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    all_preds, all_labels = [], []
    running_loss, nsamples = 0.0, 0

    with torch.no_grad():
        for images, feats, labels in test_loader:
            images, feats, labels = images.to(device), feats.to(device), labels.to(device)
            outputs = model(images, feats)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            nsamples += images.size(0)

            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / nsamples
    loss_list.append(avg_loss)

    all_preds = torch.tensor(all_preds)
    all_labels = torch.tensor(all_labels)
    nclass = len(torch.unique(all_labels))

    acc  = Accuracy(task="multiclass", num_classes=nclass)(all_preds, all_labels).item()
    prec = Precision(task="multiclass", num_classes=nclass)(all_preds, all_labels).item()
    rec  = Recall(task="multiclass", num_classes=nclass)(all_preds, all_labels).item()
    f1   = F1Score(task="multiclass", num_classes=nclass)(all_preds, all_labels).item()
    cm   = ConfusionMatrix(task="multiclass", num_classes=nclass)(all_preds, all_labels)

    acc_list.append(acc)
    prec_list.append(prec)
    rec_list.append(rec)
    f1_list.append(f1)

    fold_metrics[fold] = {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "loss": avg_loss}

    print(f"[Fold {fold}] Acurácia: {acc:.4f} | Precisão: {prec:.4f} | Recall: {rec:.4f} | Test Loss: {avg_loss:.4f}")

    # salva matriz de confusão
    plot_confusion_matrix(
        cm.cpu().numpy(),
        save_path=base_dir / f"fold_{fold}_best_model.png",
        title=f"Confusion Matrix – Fold {fold}",
    )

# -------------------------------------------------------------------------
# Estatísticas agregadas
# -------------------------------------------------------------------------
print("\n=== Estatísticas Finais ===")
mean_acc,  std_acc  = print_final_stats(acc_list,  "Acurácia")
mean_prec, std_prec = print_final_stats(prec_list, "Precisão")
mean_rec,  std_rec  = print_final_stats(rec_list,  "Recall")
mean_f1,   std_f1   = print_final_stats(f1_list,   "F1-score")
mean_loss, std_loss = print_final_stats(loss_list, "Test Loss")

k = len(acc_list) or 1  # evita div/0 se algum fold faltou

# Intervalos de confiança 95 %
ci_acc_low,  ci_acc_high  = ci95(mean_acc,  std_acc,  k)
ci_prec_low, ci_prec_high = ci95(mean_prec, std_prec, k)
ci_rec_low,  ci_rec_high  = ci95(mean_rec,  std_rec,  k)
ci_f1_low,   ci_f1_high   = ci95(mean_f1,   std_f1,   k)
ci_loss_low, ci_loss_high = ci95(mean_loss, std_loss, k)

# Estatísticas-t (H0: μ = 0)
t_acc  = t_stat(mean_acc,  std_acc,  k)
t_prec = t_stat(mean_prec, std_prec, k)
t_rec  = t_stat(mean_rec,  std_rec,  k)
t_f1   = t_stat(mean_f1,   std_f1,   k)
t_loss = t_stat(mean_loss, std_loss, k)

print("=== Estatística-t (H0: μ = 0) ===")
print(f"Acurácia: t({k-1}) = {t_acc:.4f}")
print(f"Precisão: t({k-1}) = {t_prec:.4f}")
print(f"Recall:   t({k-1}) = {t_rec:.4f}")
print(f"F1-score: t({k-1}) = {t_f1:.4f}")
print(f"TestLoss: t({k-1}) = {t_loss:.4f}")

# -------------------------------------------------------------------------
# Gravação em arquivo .txt
# -------------------------------------------------------------------------
txt_path = base_dir / f"{hyper['NAME_DATASET']}_{hyper['TMODEL']}_resultados.txt"
with open(txt_path, "w") as f:
    f.write(f"Arquivo gerado em: {datetime.now():%d/%m/%Y – %H:%M:%S}\n\n")

    for fold, m in fold_metrics.items():
        f.write(f"Fold {fold}:\n")
        f.write(f"  Acurácia: {m['acc']:.4f}\n")
        f.write(f"  Precisão: {m['prec']:.4f}\n")
        f.write(f"  Recall:   {m['rec']:.4f}\n")
        f.write(f"  F1-score: {m['f1']:.4f}\n")
        f.write(f"  Test Loss: {m['loss']:.6f}\n\n")

    f.write("=== Métricas Finais ===\n")
    f.write(f"Acurácia: Média={mean_acc:.4f}, Desv={std_acc:.4f}\n")
    f.write(f"Precisão: Média={mean_prec:.4f}, Desv={std_prec:.4f}\n")
    f.write(f"Recall:   Média={mean_rec:.4f}, Desv={std_rec:.4f}\n")
    f.write(f"F1-score: Média={mean_f1:.4f}, Desv={std_f1:.4f}\n")
    f.write(f"Test Loss: Média={mean_loss:.6f}, Desv={std_loss:.6f}\n\n")

    f.write("=== Intervalo de Confiança 95 % (t-Student) ===\n")
    f.write(f"Acurácia: [{ci_acc_low:.4f}, {ci_acc_high:.4f}]\n")
    f.write(f"Precisão: [{ci_prec_low:.4f}, {ci_prec_high:.4f}]\n")
    f.write(f"Recall:   [{ci_rec_low:.4f}, {ci_rec_high:.4f}]\n")
    f.write(f"F1-score: [{ci_f1_low:.4f}, {ci_f1_high:.4f}]\n")
    f.write(f"Test Loss: [{ci_loss_low:.6f}, {ci_loss_high:.6f}]\n\n")

    f.write("=== Estatística-t (H0: μ = 0) ===\n")
    f.write(f"Acurácia: t({k-1}) = {t_acc:.4f}\n")
    f.write(f"Precisão: t({k-1}) = {t_prec:.4f}\n")
    f.write(f"Recall:   t({k-1}) = {t_rec:.4f}\n")
    f.write(f"F1-score: t({k-1}) = {t_f1:.4f}\n")
    f.write(f"Test Loss: t({k-1}) = {t_loss:.4f}\n")

print(f"\n✓ Resultados completos salvos em {txt_path}")
