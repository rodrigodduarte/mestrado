#!/usr/bin/env python3
"""
collect_stats.py
────────────────
Varredura de diretórios de experimentos - modelos_kf/<DATASET>_<EXPNAME>
• Carrega automaticamente o melhor checkpoint de **cada fold** (fold_?_best_model*.ckpt)
• Avalia no loader de teste correspondente
• Calcula:
    – Accuracy, Precision, Recall, F1-score, Test-Loss (por fold)
    – Média, Desvio-padrão, IC-95 % (t-Student), t-estatística e p-valor (baseline µ0)
• Imprime resumo no console e grava um <DATASET>_<EXPNAME>_stats.txt
    com comentários (linha iniciada em “#”) explicando cada métrica
Destinado a arquivar resultados para análises futuras (meta-estudos, papers etc.)
"""

# ───────────────────────────── IMPORTS ────────────────────────────── #
import os, random, yaml, argparse, textwrap
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from scipy import stats
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix
import pytorch_lightning as pl

# 👉 importa suas classes
from model import CustomModel, CustomEnsembleModel, CustomModelTriple
from kf_data import CustomImageCSVModule_kf, CustomImageModule_kf

# ─────────────────────────── UTILITÁRIOS ──────────────────────────── #
def set_seeds(seed=42):
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    random.seed(seed), np.random.seed(seed)
    torch.manual_seed(seed), torch.cuda.manual_seed_all(seed)

def load_yaml(path):
    with open(path, "r") as f: return yaml.safe_load(f)

def ci95(mean, std, k):
    return stats.t.interval(0.95, df=k-1, loc=mean, scale=std/np.sqrt(k))

def t_test(mean, std, k, mu0):
    se = std / np.sqrt(k)
    t_val = (mean - mu0) / se
    p_val = stats.t.sf(abs(t_val), df=k-1) * 2
    return t_val, p_val

def pick_model_class(tmodel, with_vec=False, triple=False):
    if triple:                      return CustomModelTriple
    if with_vec:                    return CustomEnsembleModel
    return CustomModel              # backbone “puro”

# ─────────────────────────── AVALIAÇÃO ────────────────────────────── #
def evaluate_experiment(exp_dir: Path, cfg_path: Path, baseline=None):
    """Avalia todos os folds dentro de exp_dir usando o config do treino."""
    cfg = load_yaml(cfg_path)
    baseline = baseline or dict(acc=.95, prec=.95, rec=.95, f1=.95, loss=0.0)

    # determina qual classe de modelo usar
    tmodel = cfg["TMODEL"]
    triple = "triplo" in exp_dir.name or "triple" in exp_dir.name
    with_vec = ("_ne" not in exp_dir.name) and (not triple)
    ModelClass = pick_model_class(tmodel, with_vec, triple)

    accs, precs, recs, f1s, losses = ([] for _ in range(5))
    fold_metrics = {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold in range(cfg["K_FOLDS"]):
        # aceita nomes fold_X_best_model* (com ou sem -vN, .ckpt primeiro na ordem)
        ckpts = sorted(exp_dir.glob(f"fold_{fold}_best_model*.ckpt"))
        if not ckpts:
            print(f"[{exp_dir.name}] fold {fold}: checkpoint não encontrado.")
            continue
        ckpt = ckpts[0]
        model = ModelClass.load_from_checkpoint(str(ckpt)).to(device).eval()

        # módulo de dados apropriado (com vetor → CSVModule_kf; sem vetor → ImageModule_kf)
        uses_feats = with_vec or triple
        DMClass = CustomImageCSVModule_kf if uses_feats else CustomImageModule_kf
        dm = DMClass(
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
        running_loss, nsamples = 0.0, 0

        with torch.no_grad():
            for batch in loader:
                if uses_feats:
                    imgs, feats, y = batch
                    imgs, feats, y = imgs.to(device), feats.to(device), y.to(device)
                    out = model(imgs, feats)
                else:
                    imgs, y = batch
                    imgs, y = imgs.to(device), y.to(device)
                    out = model(imgs)
                running_loss += criterion(out, y).item() * imgs.size(0)
                nsamples += imgs.size(0)
                preds.extend(out.argmax(1).cpu().numpy())
                labels.extend(y.cpu().numpy())

        loss = running_loss / nsamples
        tensor_p, tensor_l = torch.tensor(preds), torch.tensor(labels)
        ncls = len(torch.unique(tensor_l))
        acc  = Accuracy(task='multiclass', num_classes=ncls)(tensor_p, tensor_l).item()
        prec = Precision(task='multiclass', num_classes=ncls)(tensor_p, tensor_l).item()
        rec  = Recall(task='multiclass', num_classes=ncls)(tensor_p, tensor_l).item()
        f1   = F1Score(task='multiclass', num_classes=ncls)(tensor_p, tensor_l).item()

        accs.append(acc); precs.append(prec); recs.append(rec); f1s.append(f1); losses.append(loss)
        fold_metrics[fold] = dict(acc=acc, prec=prec, rec=rec, f1=f1, loss=loss)
        print(f"[{exp_dir.name}] fold {fold} – acc={acc:.4f}  loss={loss:.4f}")

    # ─── Estatísticas globais ───
    k = len(accs)
    if k == 0: return  # nada foi avaliado

    def summar(v): return np.mean(v), np.std(v, ddof=1)
    m_acc, s_acc = summar(accs)
    m_prec, s_prec = summar(precs)
    m_rec, s_rec = summar(recs)
    m_f1, s_f1 = summar(f1s)
    m_loss, s_loss = summar(losses)

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

    # ─── Salvar arquivo de texto ───
    out_txt = exp_dir / f"{cfg['NAME_DATASET']}_{cfg['TMODEL']}_stats.txt"
    with open(out_txt, "w") as f:
        header = textwrap.dedent(f"""
        # Relatório de Estatísticas – {exp_dir.name}
        # Gerado em {datetime.now():%d/%m/%Y – %H:%M:%S}
        #
        #   acc, prec, rec, f1  → métricas de acurácia por fold
        #   loss                → Cross-Entropy média por fold
        #   média / desvio      → agregados amostrais dos {k} folds
        #   IC95%               → intervalo de confiança 95 % (t-Student)
        #   t, p                → teste-t bilateral contra baseline µ0 (= {baseline})
        #
        """).lstrip()
        f.write(header)
        for fold, m in fold_metrics.items():
            f.write(f"Fold {fold}:  acc={m['acc']:.4f}  prec={m['prec']:.4f} "
                    f"rec={m['rec']:.4f}  f1={m['f1']:.4f}  loss={m['loss']:.6f}\n")
        f.write("\n# --- Resumo ---\n")
        def line(lbl, m, s, ci, t, p):
            f.write(f"{lbl:<8} média={m:.4f}  desv={s:.4f}  "
                    f"IC95%=[{ci[0]:.4f},{ci[1]:.4f}]  t={t:.2f}  p={p:.2e}\n")
        line("Acc",  m_acc,  s_acc,  ci_acc,  t_acc,  p_acc)
        line("Prec", m_prec, s_prec, ci_prec, t_prec, p_prec)
        line("Rec",  m_rec,  s_rec,  ci_rec,  t_rec,  p_rec)
        line("F1",   m_f1,   s_f1,   ci_f1,   t_f1,   p_f1)
        line("Loss", m_loss, s_loss, ci_loss, t_loss, p_loss)
    print(f"✓ Estatísticas salvas em {out_txt}\n")

# ──────────────────────────── MAIN CLI ────────────────────────────── #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extrai estatísticas de modelos treinados (k-fold).")
    parser.add_argument("experiments", nargs="+",
        help="Caminhos de diretórios de experimento dentro de modelos_kf/")
    parser.add_argument("--config", default="config.yaml",
        help="Arquivo .yaml com hiperparâmetros (usado para caminhos de dados)")
    args = parser.parse_args()

    set_seeds()

    for exp in args.experiments:
        exp_dir = Path(exp)
        if not exp_dir.exists():
            print(f"⚠️  {exp_dir} não encontrado – pulando.")
            continue
        evaluate_experiment(exp_dir, Path(args.config))
