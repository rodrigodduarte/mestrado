#!/usr/bin/env python3
"""
estatisticas_ne.py  –  avalia modelos *sem ensemble* (CustomModel) salvos em
modelos_kf/<DATASET>_<TMODEL>_ne/, calcula a estatística-t sobre a média dos
folds e grava um resumo .txt.

Correção: usa model.num_classes em vez de model.fc.out_features.
"""

import os, random, yaml, argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from scipy import stats
from torchmetrics import Accuracy, Precision, Recall, F1Score

from model import CustomModel                      # modelo base
from kf_data import CustomImageModule_kf           # datamodule só com imagens


# ---------- helpers ----------
def set_seeds(seed=42):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def load_yaml(path):                 # config2.yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)

def t_val(mean, std, k, mu0):
    se = std / np.sqrt(k)
    t_ = (mean - mu0) / se
    p_ = stats.t.sf(abs(t_), df=k-1) * 2
    return t_, p_
# --------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="config2.yaml")
    args = ap.parse_args()

    cfg = load_yaml(args.cfg)
    set_seeds()

    base = Path("modelos_kf") / f"{cfg['NAME_DATASET']}_{cfg['TMODEL']}_ne"
    if not base.exists():
        raise SystemExit(f"Pasta {base} não encontrada.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    accs, precs, recs, f1s, losses = ([] for _ in range(5))

    for fold in range(cfg["K_FOLDS"]):
        ckpt = next(iter(sorted(base.glob(f"fold_{fold}_best_model*.ckpt"))), None)
        if ckpt is None:
            print(f"Fold {fold}: checkpoint ausente – pulando.")
            continue

        model = CustomModel.load_from_checkpoint(str(ckpt)).to(device).eval()
        n_classes = model.num_classes                     # ← uso genérico

        dm = CustomImageModule_kf(
            train_dir=cfg["TRAIN_DIR"],
            test_dir=cfg["TEST_DIR"],
            shape=cfg["SHAPE"],
            batch_size=cfg["BATCH_SIZE"],
            num_workers=cfg["NUM_WORKERS"],
            n_splits=cfg["K_FOLDS"],
            fold_idx=fold,
        )
        dm.setup("test")
        loader = dm.test_dataloader()

        criterion = torch.nn.CrossEntropyLoss()
        preds, labels = [], []
        tot_loss, nsamp = 0.0, 0

        with torch.no_grad():
            for imgs, y in loader:
                imgs, y = imgs.to(device), y.to(device)

                # shift opcional: CSV com labels 1…N  → 0…N-1
                if y.min() == 1 and y.max() == n_classes:
                    y = y - 1

                out = model(imgs)

                if y.max() >= n_classes:
                    raise RuntimeError(
                        f"[Fold {fold}] label fora de faixa: y.max={y.max().item()} "
                        f">= n_classes={n_classes}")

                tot_loss += criterion(out, y).item() * imgs.size(0)
                nsamp += imgs.size(0)
                preds.extend(out.argmax(1).cpu().numpy())
                labels.extend(y.cpu().numpy())

        loss = tot_loss / nsamp
        tp, tl = torch.tensor(preds), torch.tensor(labels)

        acc  = Accuracy(task="multiclass", num_classes=n_classes)(tp, tl).item()
        prec = Precision(task="multiclass", num_classes=n_classes)(tp, tl).item()
        rec  = Recall(task="multiclass", num_classes=n_classes)(tp, tl).item()
        f1   = F1Score(task="multiclass", num_classes=n_classes)(tp, tl).item()

        accs.append(acc); precs.append(prec); recs.append(rec); f1s.append(f1); losses.append(loss)
        print(f"[Fold {fold}] Acc={acc:.4f}  Loss={loss:.4f}")

    k = len(accs)
    if k == 0:
        raise SystemExit("Nenhum fold avaliado.")

    m_acc,  s_acc  = np.mean(accs),  np.std(accs,  ddof=1)
    m_prec, s_prec = np.mean(precs), np.std(precs, ddof=1)
    m_rec,  s_rec  = np.mean(recs),  np.std(recs,  ddof=1)
    m_f1,   s_f1   = np.mean(f1s),   np.std(f1s,   ddof=1)
    m_loss, s_loss = np.mean(losses),np.std(losses,ddof=1)

    t_acc,  p_acc  = t_val(m_acc,  s_acc,  k, 0.95)
    t_prec, p_prec = t_val(m_prec, s_prec, k, 0.95)
    t_rec,  p_rec  = t_val(m_rec,  s_rec,  k, 0.95)
    t_f1,   p_f1   = t_val(m_f1,   s_f1,   k, 0.95)
    t_loss, p_loss = t_val(m_loss, s_loss, k, 0.0)

    print("\n=== Estatística-t (bilateral) ===")
    print(f"Acurácia: t({k-1}) = {t_acc:.2f},  p = {p_acc:.2e}")
    print(f"Precisão: t({k-1}) = {t_prec:.2f}, p = {p_prec:.2e}")
    print(f"Recall:   t({k-1}) = {t_rec:.2f},  p = {p_rec:.2e}")
    print(f"F1-score: t({k-1}) = {t_f1:.2f},   p = {p_f1:.2e}")
    print(f"Loss:     t({k-1}) = {t_loss:.2f}, p = {p_loss:.2e}")

    out_txt = base / f"{cfg['NAME_DATASET']}_{cfg['TMODEL']}_ne_resultados.txt"
    with open(out_txt, "w") as f:
        f.write(f"# Estatística-t gerada em {datetime.now():%d/%m/%Y %H:%M:%S}\n")
        f.write(f"Acurácia t={t_acc:.2f} p={p_acc:.2e}\n")
        f.write(f"Precisão t={t_prec:.2f} p={p_prec:.2e}\n")
        f.write(f"Recall   t={t_rec:.2f} p={p_rec:.2e}\n")
        f.write(f"F1-score t={t_f1:.2f} p={p_f1:.2e}\n")
        f.write(f"Loss     t={t_loss:.2f} p={p_loss:.2e}\n")
    print(f"\n✓ Resultado salvo em {out_txt}")

if __name__ == "__main__":
    main()
