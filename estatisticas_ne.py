#!/usr/bin/env python3
"""
estatisticas_ne.py
──────────────────
Avalia **modelos base (sem ensemble nem vetor de características)** treinados em
k-fold, calcula métricas por fold e imprime apenas a estatística-t (média ± desvio
dos folds).  Gera também um arquivo
    modelos_kf/<DATASET>_<TMODEL>_ne/<DATASET>_<TMODEL>_ne_resultados.txt
com o mesmo resumo.

Diferenças em relação ao estatisticas.py “com ensemble”
  • Usa CustomModel  (backbone puro)
  • Usa CustomImageModule_kf  (somente imagens)
  • Procura checkpoints na pasta *_ne
"""

# ───────────────────────── IMPORTS ────────────────────────── #
import os, random, yaml, argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from scipy import stats
from torchmetrics import Accuracy, Precision, Recall, F1Score

# ------------- módulos do projeto (ajuste se mudar o nome) -------------
from model import CustomModel
from kf_data import CustomImageModule_kf
# ----------------------------------------------------------------------- #

# ------------------- utilidades rápidas -------------------------------- #
def set_seeds(seed=42):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def load_yaml(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def t_from_mean_std(mean, std, k, mu0=0.95):
    """t (bilateral) comparando a média dos folds a µ₀."""
    se = std / np.sqrt(k)
    t_val = (mean - mu0) / se
    p_val = stats.t.sf(abs(t_val), df=k-1) * 2
    return t_val, p_val
# ----------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Avalia checkpoints *_ne (sem ensemble) e imprime t-teste.")
    parser.add_argument("--cfg", default="config.yaml",
                        help="Arquivo de hiperparâmetros do treino")
    args = parser.parse_args()

    cfg = load_yaml(args.cfg)
    set_seeds()

    # ---- pasta onde estão os checkpoints sem ensemble ----
    base_dir = Path("modelos_kf") / f"{cfg['NAME_DATASET']}_{cfg['TMODEL']}_ne"
    if not base_dir.exists():
        raise SystemExit(f"{base_dir} não encontrado.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    accs, precs, recs, f1s, losses = ([] for _ in range(5))

    for fold in range(cfg["K_FOLDS"]):
        ckpt_list = sorted(base_dir.glob(f"fold_{fold}_best_model*.ckpt"))
        if not ckpt_list:
            print(f"⚠️  Fold {fold}: checkpoint ausente, pulando.")
            continue

        model = CustomModel.load_from_checkpoint(str(ckpt_list[0])).to(device).eval()

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
        running_loss, n = 0.0, 0

        with torch.no_grad():
            for imgs, y in loader:
                imgs, y = imgs.to(device), y.to(device)

                # rótulos 1-based?  converte para 0-based se necessário
                if y.min() == 1 and y.max() == model.fc.out_features:
                    y = y - 1

                if y.max() >= model.fc.out_features:
                    raise RuntimeError(
                        f"Label fora de faixa no fold {fold}: y.max={y.max().item()} "
                        f"≥ n_classes={model.fc.out_features}")

                out = model(imgs)
                running_loss += criterion(out, y).item() * imgs.size(0)
                n += imgs.size(0)
                preds.extend(out.argmax(1).cpu().numpy())
                labels.extend(y.cpu().numpy())

        loss = running_loss / n
        tp, tl = torch.tensor(preds), torch.tensor(labels)
        ncls = len(torch.unique(tl))

        acc  = Accuracy(task="multiclass", num_classes=ncls)(tp, tl).item()
        prec = Precision(task="multiclass", num_classes=ncls)(tp, tl).item()
        rec  = Recall(task="multiclass", num_classes=ncls)(tp, tl).item()
        f1   = F1Score(task="multiclass", num_classes=ncls)(tp, tl).item()

        accs.append(acc); precs.append(prec); recs.append(rec); f1s.append(f1); losses.append(loss)
        print(f"[Fold {fold}] Acc={acc:.4f}  Loss={loss:.4f}")

    k = len(accs)
    if k == 0:
        raise SystemExit("Nenhum fold avaliado.")

    # ─── média / desvio ──
    m_acc,  s_acc  = np.mean(accs),  np.std(accs,  ddof=1)
    m_prec, s_prec = np.mean(precs), np.std(precs, ddof=1)
    m_rec,  s_rec  = np.mean(recs),  np.std(recs,  ddof=1)
    m_f1,   s_f1   = np.mean(f1s),   np.std(f1s,   ddof=1)
    m_loss, s_loss = np.mean(losses),np.std(losses,ddof=1)

    # ─── t-test (baseline µ₀ = 0.95 p/ acurácia, 0 p/ loss) ──
    t_acc,  p_acc  = t_from_mean_std(m_acc,  s_acc,  k, mu0=0.95)
    t_prec, p_prec = t_from_mean_std(m_prec, s_prec, k, mu0=0.95)
    t_rec,  p_rec  = t_from_mean_std(m_rec,  s_rec,  k, mu0=0.95)
    t_f1,   p_f1   = t_from_mean_std(m_f1,   s_f1,   k, mu0=0.95)
    t_loss, p_loss = t_from_mean_std(m_loss, s_loss, k, mu0=0.0)

    # ─── imprime apenas o teste-t ──
    print("\n=== Estatística-t (bilateral) ===")
    print(f"Acurácia: t({k-1}) = {t_acc:.2f},  p = {p_acc:.2e}")
    print(f"Precisão: t({k-1}) = {t_prec:.2f}, p = {p_prec:.2e}")
    print(f"Recall:   t({k-1}) = {t_rec:.2f},  p = {p_rec:.2e}")
    print(f"F1-score: t({k-1}) = {t_f1:.2f},   p = {p_f1:.2e}")
    print(f"Loss:     t({k-1}) = {t_loss:.2f}, p = {p_loss:.2e}")

    # ─── salva txt com o mesmo resumo ──
    out_txt = base_dir / f"{cfg['NAME_DATASET']}_{cfg['TMODEL']}_ne_resultados.txt"
    with open(out_txt, "w") as f:
        f.write(f"# Estatística-t – gerado em {datetime.now():%d/%m/%Y %H:%M:%S}\n")
        f.write(f"# k = {k} folds, baseline µ₀ = 0.95 (accuracy) / 0.0 (loss)\n")
        f.write(f"Acurácia t={t_acc:.2f} p={p_acc:.2e}\n")
        f.write(f"Precisão t={t_prec:.2f} p={p_prec:.2e}\n")
        f.write(f"Recall   t={t_rec:.2f} p={p_rec:.2e}\n")
        f.write(f"F1-score t={t_f1:.2f} p={p_f1:.2e}\n")
        f.write(f"Loss     t={t_loss:.2f} p={p_loss:.2e}\n")
    print(f"\n✓ Resultado salvo em {out_txt}")

if __name__ == "__main__":
    main()
