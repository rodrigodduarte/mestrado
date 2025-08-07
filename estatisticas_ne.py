#!/usr/bin/env python3
"""
estatisticas.py
Avalia os checkpoints de cada fold, calcula métricas, intervalos de confiança
(95 % t-Student) e estatísticas-t contra um baseline configurável.
Salva tudo em <dataset>_<modelo>_resultados.txt dentro da pasta modelos_kf/.
"""

# -------------------------------------------------------------------- #
# Imports
# -------------------------------------------------------------------- #
import os, random, yaml
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from model import CustomModel
from kf_data import CustomImageCSVModule_kf

# -------------------------------------------------------------------- #
# Configurações do teste t
# -------------------------------------------------------------------- #
MU0 = {           # hipótese nula (μ0) para cada métrica
    "acc":  0.95,
    "prec": 0.95,
    "rec":  0.95,
    "f1":   0.95,
    "loss": 0.0,   # se quiser testar perda média > 0.01, mude aqui
}

# -------------------------------------------------------------------- #
# Funções utilitárias
# -------------------------------------------------------------------- #
def set_seeds(seed=42):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_cfg(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def plot_confmat(cm, path, title):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predito"); plt.ylabel("Real"); plt.title(title)
    plt.tight_layout()
    plt.savefig(path); plt.close()


def print_stats(values, name):
    arr = np.array(values)
    mean, std = arr.mean(), arr.std(ddof=1)
    print(f"{name} por Fold: {arr}")
    print(f"{name} Média: {mean:.4f} | Desvio Padrão: {std:.4f}\n")
    return mean, std


def ci95(mean, std, k):
    return stats.t.interval(0.95, df=k-1, loc=mean, scale=std/np.sqrt(k))


def t_test(mean, std, k, mu0):
    se = std / np.sqrt(k)
    t_stat = (mean - mu0) / se
    p_val = stats.t.sf(abs(t_stat), df=k-1) * 2   # bilateral
    return t_stat, p_val


# -------------------------------------------------------------------- #
# Pipeline
# -------------------------------------------------------------------- #
def main():
    set_seeds()
    cfg = load_cfg()

    base = Path("modelos_kf") / f"{cfg['NAME_DATASET']}_{cfg['TMODEL']}"
    base.mkdir(parents=True, exist_ok=True)

    # coletores por fold
    accs, precs, recs, f1s, losses = ([] for _ in range(5))
    fold_metrics = {}

    for fold in range(cfg["K_FOLDS"]):
        ckpt = base / f"fold_{fold}_best_model.ckpt"
        if not ckpt.exists():
            print(f"[Fold {fold}] checkpoint ausente → pulando.")
            continue

        print(f"[Fold {fold}] Avaliando {ckpt}")
        model = CustomModel.load_from_checkpoint(str(ckpt), scale_factor=1.0)
        model.eval().to("cuda" if torch.cuda.is_available() else "cpu")

        dm = CustomImageCSVModule_kf(
            train_dir=cfg["TRAIN_DIR"],
            test_dir=cfg["TEST_DIR"],
            shape=cfg["SHAPE"],
            batch_size=cfg["BATCH_SIZE"],
            num_workers=cfg["NUM_WORKERS"],
            n_splits=cfg["K_FOLDS"],
            fold_idx=fold,
        )
        dm.setup("test")
        test_loader = dm.test_dataloader()

        criterion = torch.nn.CrossEntropyLoss()
        preds, labels = [], []
        tot_loss, n = 0.0, 0

        with torch.no_grad():
            for imgs, feats, y in test_loader:
                imgs, feats, y = imgs.cuda(), feats.cuda(), y.cuda() if torch.cuda.is_available() else (imgs, feats, y)
                out = model(imgs, feats)
                tot_loss += criterion(out, y).item() * imgs.size(0)
                n += imgs.size(0)
                preds.extend(out.argmax(1).cpu().numpy())
                labels.extend(y.cpu().numpy())

        avg_loss = tot_loss / n
        tensor_p, tensor_l = torch.tensor(preds), torch.tensor(labels)
        ncls = len(torch.unique(tensor_l))

        acc  = Accuracy(task="multiclass", num_classes=ncls)(tensor_p, tensor_l).item()
        prec = Precision(task="multiclass", num_classes=ncls)(tensor_p, tensor_l).item()
        rec  = Recall(task="multiclass", num_classes=ncls)(tensor_p, tensor_l).item()
        f1   = F1Score(task="multiclass", num_classes=ncls)(tensor_p, tensor_l).item()
        cm   = ConfusionMatrix(task="multiclass", num_classes=ncls)(tensor_p, tensor_l)

        accs.append(acc); precs.append(prec); recs.append(rec); f1s.append(f1); losses.append(avg_loss)
        fold_metrics[fold] = dict(acc=acc, prec=prec, rec=rec, f1=f1, loss=avg_loss)

        plot_confmat(cm.cpu().numpy(), base / f"fold_{fold}_matrix.png", f"Fold {fold}")

        print(f"[Fold {fold}] Acc={acc:.4f} | Prec={prec:.4f} | Rec={rec:.4f} | Loss={avg_loss:.4f}")

    # ------------------------------------------------------------------ #
    # Estatísticas finais
    # ------------------------------------------------------------------ #
    k = len(accs)
    print("\n=== Estatísticas Finais ===")
    m_acc,  s_acc  = print_stats(accs,  "Acurácia")
    m_prec, s_prec = print_stats(precs, "Precisão")
    m_rec,  s_rec  = print_stats(recs,  "Recall")
    m_f1,   s_f1   = print_stats(f1s,   "F1-score")
    m_loss, s_loss = print_stats(losses,"Test Loss")

    # IC 95 %
    ci_acc  = ci95(m_acc,  s_acc,  k)
    ci_prec = ci95(m_prec, s_prec, k)
    ci_rec  = ci95(m_rec,  s_rec,  k)
    ci_f1   = ci95(m_f1,   s_f1,   k)
    ci_loss = ci95(m_loss, s_loss, k)

    # Estatística-t e p-valor
    t_acc,  p_acc  = t_test(m_acc,  s_acc,  k, MU0["acc"])
    t_prec, p_prec = t_test(m_prec, s_prec, k, MU0["prec"])
    t_rec,  p_rec  = t_test(m_rec,  s_rec,  k, MU0["rec"])
    t_f1,   p_f1   = t_test(m_f1,   s_f1,   k, MU0["f1"])
    t_loss, p_loss = t_test(m_loss, s_loss, k, MU0["loss"])

    print("=== Estatística-t (bilateral) ===")
    print(f"Acurácia: t({k-1}) = {t_acc:.2f},  p = {p_acc:.2e}")
    print(f"Precisão: t({k-1}) = {t_prec:.2f}, p = {p_prec:.2e}")
    print(f"Recall:   t({k-1}) = {t_rec:.2f},  p = {p_rec:.2e}")
    print(f"F1-score: t({k-1}) = {t_f1:.2f},   p = {p_f1:.2e}")
    print(f"Loss:     t({k-1}) = {t_loss:.2f}, p = {p_loss:.2e}")

    # ------------------------------------------------------------------ #
    # Salva em .txt
    # ------------------------------------------------------------------ #
    txt = base / f"{cfg['NAME_DATASET']}_{cfg['TMODEL']}_resultados.txt"
    with open(txt, "w") as f:
        f.write(f"Arquivo gerado em: {datetime.now():%d/%m/%Y – %H:%M:%S}\n\n")
        for fold, m in fold_metrics.items():
            f.write(f"Fold {fold}:\n")
            f.write(f"  Acc  = {m['acc']:.4f}\n")
            f.write(f"  Prec = {m['prec']:.4f}\n")
            f.write(f"  Rec  = {m['rec']:.4f}\n")
            f.write(f"  F1   = {m['f1']:.4f}\n")
            f.write(f"  Loss = {m['loss']:.6f}\n\n")

        def w(title, mean, std, ci, t, p):
            f.write(f"{title}: Média={mean:.4f}, Desv={std:.4f}, "
                    f"IC95%=[{ci[0]:.4f},{ci[1]:.4f}], "
                    f"t({k-1})={t:.2f}, p={p:.2e}\n")

        f.write("=== Resumo ===\n")
        w("Acurácia", m_acc, s_acc, ci_acc, t_acc, p_acc)
        w("Precisão", m_prec, s_prec, ci_prec, t_prec, p_prec)
        w("Recall  ", m_rec, s_rec, ci_rec, t_rec, p_rec)
        w("F1-score", m_f1, s_f1, ci_f1, t_f1, p_f1)
        w("Loss    ", m_loss, s_loss, ci_loss, t_loss, p_loss)

    print(f"\n✓ Resultados completos salvos em {txt}")


if __name__ == "__main__":
    main()
