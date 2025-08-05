#!/usr/bin/env python3
"""
collect_stats.py
Extrai métricas fold-a-fold de cada experimento (modelos_kf/<DIR>)
e gera um  <DATASET>_<EXPNAME>_stats.txt  com comentários (#).
Corrigido para datasets cujos rótulos começam em 1 (converter p/ 0-based).
"""

import os, random, yaml, argparse, textwrap
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from scipy import stats
from torchmetrics import Accuracy, Precision, Recall, F1Score

# ── imports do seu projeto ───────────────────────────────────────────
from model import CustomModel, CustomEnsembleModel, CustomModelTriple
from kf_data import CustomImageCSVModule_kf, CustomImageModule_kf
# ─────────────────────────────────────────────────────────────────────

# ---------- utilidades ----------
def set_seeds(seed=42):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def load_yaml(path):  # config.yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ci95(mean, std, k):
    return stats.t.interval(0.95, df=k-1, loc=mean, scale=std/np.sqrt(k))

def t_test(mean, std, k, mu0):
    se = std / np.sqrt(k)
    t_val = (mean - mu0) / se
    p_val = stats.t.sf(abs(t_val), df=k-1) * 2
    return t_val, p_val

def pick_model(tmodel, with_vec=False, triple=False):
    if triple:            return CustomModelTriple
    if with_vec:          return CustomEnsembleModel
    return CustomModel

# ---------- avaliação ----------
def evaluate_experiment(exp_dir: Path,
                        cfg_path: Path,
                        baseline=None):
    cfg = load_yaml(cfg_path)
    baseline = baseline or dict(acc=.95, prec=.95, rec=.95, f1=.95, loss=0.0)

    tmodel   = cfg["TMODEL"]
    triple   = "triple" in exp_dir.name or "triplo" in exp_dir.name
    with_vec = ("_ne" not in exp_dir.name) and (not triple)
    ModelCls = pick_model(tmodel, with_vec, triple)
    uses_feat = with_vec or triple
    DMCls = CustomImageCSVModule_kf if uses_feat else CustomImageModule_kf

    accs, precs, recs, f1s, losses = ([] for _ in range(5))
    fold_metrics = {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold in range(cfg["K_FOLDS"]):
        ckpts = sorted(exp_dir.glob(f"fold_{fold}_best_model*.ckpt"))
        if not ckpts:
            print(f"[{exp_dir.name}] fold {fold}: checkpoint ausente.")
            continue
        model = ModelCls.load_from_checkpoint(str(ckpts[0])).to(device).eval()

        dm = DMCls(
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
        tot_loss, n = 0.0, 0

        with torch.no_grad():
            for batch in loader:
                if uses_feat:
                    imgs, feats, y = batch
                    imgs, feats, y = imgs.to(device), feats.to(device), y.to(device)
                    out = model(imgs, feats)
                else:
                    imgs, y = batch
                    imgs, y = imgs.to(device), y.to(device)
                    out = model(imgs)

                # —— CORREÇÃO 1-based → 0-based ——
                if y.min() == 1:
                    y = y - 1

                tot_loss += criterion(out, y).item() * imgs.size(0)
                n += imgs.size(0)
                preds.extend(out.argmax(1).cpu().numpy())
                labels.extend(y.cpu().numpy())

        loss = tot_loss / n
        tp, tl = torch.tensor(preds), torch.tensor(labels)
        ncls   = len(torch.unique(tl))

        acc  = Accuracy(task="multiclass", num_classes=ncls)(tp, tl).item()
        prec = Precision(task="multiclass", num_classes=ncls)(tp, tl).item()
        rec  = Recall(task="multiclass", num_classes=ncls)(tp, tl).item()
        f1   = F1Score(task="multiclass", num_classes=ncls)(tp, tl).item()

        accs.append(acc); precs.append(prec); recs.append(rec); f1s.append(f1); losses.append(loss)
        fold_metrics[fold] = dict(acc=acc, prec=prec, rec=rec, f1=f1, loss=loss)
        print(f"[{exp_dir.name}] fold {fold}: acc={acc:.4f} loss={loss:.4f}")

    # —— agregados ——
    k = len(accs)
    if k == 0:
        print(f"[{exp_dir.name}] nenhum fold avaliado.")
        return

    def mean_std(arr): return np.mean(arr), np.std(arr, ddof=1)
    m_acc,  s_acc  = mean_std(accs)
    m_prec, s_prec = mean_std(precs)
    m_rec,  s_rec  = mean_std(recs)
    m_f1,   s_f1   = mean_std(f1s)
    m_loss, s_loss = mean_std(losses)

    ci_acc  = ci95(m_acc,  s_acc,  k)
    ci_prec = ci95(m_prec, s_prec, k)
    ci_rec  = ci95(m_rec,  s_rec,  k)
    ci_f1   = ci95(m_f1,   s_f1,   k)
    ci_loss = ci95(m_loss, s_loss, k)

    t_acc,  p_acc  = t_test(m_acc,  s_acc,  k, baseline["acc"])
    t_prec, p_prec = t_test(m_prec, s_prec, k, baseline["prec"])
    t_rec,  p_rec  = t_test(m_rec,  s_rec,  k, baseline["rec"])
    t_f1,   p_f1   = t_test(m_f1,   s_f1,   k, baseline["f1"])
    t_loss, p_loss = t_test(m_loss, s_loss, k, baseline["loss"])

    # —— grava arquivo txt ——
    txt = exp_dir / f"{cfg['NAME_DATASET']}_{cfg['TMODEL']}_stats.txt"
    with open(txt, "w") as f:
        f.write(textwrap.dedent(f"""
        # Estatísticas – {exp_dir.name}
        # Criado em {datetime.now():%d/%m/%Y – %H:%M:%S}
        # colunas: média / desvio / IC95% / t / p
        """).lstrip())
        for fold, m in fold_metrics.items():
            f.write(f"Fold {fold}: acc={m['acc']:.4f} prec={m['prec']:.4f} "
                    f"rec={m['rec']:.4f} f1={m['f1']:.4f} loss={m['loss']:.6f}\n")
        def row(lbl, m, s, ci, t, p):
            f.write(f"{lbl:<5} média={m:.4f} desv={s:.4f} "
                    f"IC95=[{ci[0]:.4f},{ci[1]:.4f}] t={t:.2f} p={p:.2e}\n")
        row("Acc",  m_acc,  s_acc,  ci_acc,  t_acc,  p_acc)
        row("Prec", m_prec, s_prec, ci_prec, t_prec, p_prec)
        row("Rec",  m_rec,  s_rec,  ci_rec,  t_rec,  p_rec)
        row("F1",   m_f1,   s_f1,   ci_f1,   t_f1,   p_f1)
        row("Loss", m_loss, s_loss, ci_loss, t_loss, p_loss)
    print(f"✓ Estatísticas salvas em {txt}\n")

# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extrai métricas de experimentos k-fold.")
    parser.add_argument("experiments", nargs="+",
                        help="Pastas dentro de modelos_kf/ a serem avaliadas.")
    parser.add_argument("--config", default="config.yaml",
                        help="Caminho do config.yaml do treino.")
    args = parser.parse_args()
    set_seeds()
    for exp in args.experiments:
        evaluate_experiment(Path(exp), Path(args.config))
