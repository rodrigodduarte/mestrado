#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
estatisticas_ne.py — versão adaptada para o layout seed/fold do train_kf_db.py

O que faz:
- Descobre automaticamente quantas `seed_*` existem dentro do diretório do modelo.
- Lê `seed_*/fold_*/resultados_fold.txt`, extrai `test_metrics_json` e agrega por fold.
- Para cada fold, calcula a média (e DP) das métricas entre seeds.
- Gera um arquivo `<dataset>_<tmodel>_resultados.txt` no diretório do modelo,
  com blocos "Fold i" (médias entre seeds) e "=== Métricas Finais ===" (média/DP entre folds).
- Também resume tempos: treino, teste, inferência por amostra e throughput.
- Não depende de número fixo de seeds nem de nomes exatos de chaves das métricas,
  usando heurísticas seguras para procurar accuracy/precision/recall/f1/loss.

Uso:
  python estatisticas_ne.py --run_dir modelos_kf/Flavia_convnext_t
  # ou apenas dentro do repo, se preferir:
  python estatisticas_ne.py  (e ele tenta detectar um único run_dir válido)

Requisitos de layout (produzido por train_kf_db.py):
  modelos_kf/<NAME_DATASET>_<TMODEL>/
    ├─ hyperparams_used.json
    ├─ seed_42/
    │    ├─ fold_0/resultados_fold.txt
    │    ├─ fold_1/resultados_fold.txt
    │    └─ ...
    ├─ seed_43/...
    └─ ...

Saída:
  modelos_kf/<NAME_DATASET>_<TMODEL>/<NAME_DATASET>_<TMODEL>_resultados.txt
"""

from __future__ import annotations
import json, math, re
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Any, List, Tuple, Optional

# ------------------------- Utilidades -------------------------

def _safe_load_json(s: str) -> Optional[dict]:
    try:
        return json.loads(s)
    except Exception:
        return None

def _first_key(d: Dict[str, Any], patterns: List[str]) -> Optional[str]:
    """Retorna a primeira chave de d que casa com qualquer padrão (regex, case-insensitive)."""
    if not isinstance(d, dict):
        return None
    keys = list(d.keys())
    for pat in patterns:
        rx = re.compile(pat, flags=re.IGNORECASE)
        for k in keys:
            if rx.search(k):
                return k
    return None

def _extract_metrics(test_dict: Dict[str, Any]) -> Dict[str, float]:
    """
    Extrai métricas canônicas (acc, prec, rec, f1, loss) de um dicionário de métricas de teste.
    Usa padrões flexíveis para cobrir diferentes nomes de chaves.
    """
    if not isinstance(test_dict, dict):
        return {}

    patterns = {
        "acc":   [r"\bacc\b", r"accuracy"],
        "prec":  [r"\bprec", r"precision"],
        "rec":   [r"\brec(?!order)", r"recall"],
        "f1":    [r"\bf1\b", r"f1[_\- ]?score"],
        "loss":  [r"\bloss\b", r"test[_\- ]?loss", r"criterion"],
    }

    out = {}
    for name, pats in patterns.items():
        k = _first_key(test_dict, pats)
        if k is not None:
            try:
                out[name] = float(test_dict[k])
            except Exception:
                pass
    return out

def _extract_times(lines: List[str]) -> Dict[str, float]:
    """
    Lê tempos salvos em resultados_fold.txt pelo train_kf_db.py.
    """
    kv = {}
    for ln in lines:
        if ":" not in ln:
            continue
        k, v = ln.split(":", 1)
        k = k.strip().lower()
        v = v.strip()
        for name, keypat in [
            ("train_time_sec", r"^train_time_sec$"),
            ("test_time_sec", r"^test_time_sec$"),
            ("ms_per_sample", r"^test_inf_ms_per_sample$"),
            ("throughput",    r"^throughput_samples_per_sec$"),
        ]:
            if re.match(keypat, k):
                try:
                    kv[name] = float(v)
                except Exception:
                    pass
    return kv

def _read_resultados_fold(txt_path: Path) -> Tuple[Optional[Dict[str,float]], Optional[Dict[str,float]]]:
    """
    Retorna (metrics, times) onde metrics tem acc/prec/rec/f1/loss e times tem tempos.
    """
    try:
        s = txt_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None, None
    lines = [ln.strip() for ln in s.splitlines()]

    test_json = None
    for ln in lines:
        if ln.lower().startswith("test_metrics_json:"):
            _, js = ln.split(":", 1)
            test_json = _safe_load_json(js.strip())
            if test_json:
                break

    metrics = _extract_metrics(test_json or {})
    times = _extract_times(lines)
    return (metrics or None, times or None)

# ------------------------- Agregação -------------------------

def _discover_run_dir(run_dir: Optional[Path]) -> Path:
    """
    Se run_dir não for fornecido, tenta encontrar exatamente um diretório em modelos_kf/*_*
    que contenha pastas seed_*. Se houver mais de um, lança erro para evitar ambiguidade.
    """
    if run_dir:
        return run_dir

    base = Path("modelos_kf")
    if not base.exists():
        raise FileNotFoundError("Não encontrei a pasta 'modelos_kf'. Passe --run_dir explicitamente.")

    candidates = []
    for d in base.iterdir():
        if not d.is_dir():
            continue
        if "_" not in d.name:
            continue
        if any((d / s).is_dir() for s in d.glob("seed_*")):
            candidates.append(d)

    if len(candidates) == 0:
        raise FileNotFoundError("Nenhum diretório de execução encontrado em modelos_kf/*_*. Passe --run_dir.")
    if len(candidates) > 1:
        names = ", ".join(c.name for c in candidates[:6])
        raise RuntimeError(f"Mais de um diretório candidato encontrado ({names}). Passe --run_dir.")

    return candidates[0]

def _name_from_run_dir(run_dir: Path) -> Tuple[str, str]:
    """
    A partir de <NAME_DATASET>_<TMODEL>, retorna (dataset, tmodel).
    """
    name = run_dir.name
    if "_" not in name:
        return name, "model"
    # separar só no primeiro "_", pois modelos podem conter "_"
    parts = name.split("_", 1)
    return parts[0], parts[1]

def _collect_all(run_dir: Path) -> Tuple[Dict[int, List[Dict[str,float]]], Dict[int, List[Dict[str,float]]]]:
    """
    Percorre seed_*/fold_*/resultados_fold.txt.
    Retorna:
      - metrics_by_fold: {fold_idx: [metrics_dict_por_seed]}
      - times_by_fold:   {fold_idx: [times_dict_por_seed]}
    """
    metrics_by_fold: Dict[int, List[Dict[str,float]]] = {}
    times_by_fold: Dict[int, List[Dict[str,float]]] = {}

    seed_dirs = sorted([d for d in run_dir.glob("seed_*") if d.is_dir()],
                       key=lambda p: (len(p.name), p.name))
    if not seed_dirs:
        raise FileNotFoundError(f"Nenhuma pasta 'seed_*' encontrada em {run_dir}")

    for sd in seed_dirs:
        for fold_dir in sorted(sd.glob("fold_*")):
            if not fold_dir.is_dir():
                continue
            m = re.search(r"fold_(\d+)", fold_dir.name)
            if not m:
                continue
            fold_idx = int(m.group(1))
            txt_path = fold_dir / "resultados_fold.txt"
            if not txt_path.exists():
                # ignora folds inexistentes nessa seed
                continue

            metrics, times = _read_resultados_fold(txt_path)
            if metrics:
                metrics_by_fold.setdefault(fold_idx, []).append(metrics)
            if times:
                times_by_fold.setdefault(fold_idx, []).append(times)

    if not metrics_by_fold:
        raise RuntimeError("Não foi possível extrair métricas de nenhum resultados_fold.txt")

    return metrics_by_fold, times_by_fold

def _agg_mean_std(items: List[float]) -> Tuple[float, float]:
    if not items:
        return float("nan"), float("nan")
    if len(items) == 1:
        return items[0], 0.0
    return (mean(items), pstdev(items))

def _format_float(x: float, nd: int = 6) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "nan"
    fmt = f"{{:.{nd}f}}"
    return fmt.format(x)

# ------------------------- Principal -------------------------

def main(argv: Optional[List[str]] = None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Agrega métricas (seeds/folds) e gera <dataset>_<tmodel>_resultados.txt")
    ap.add_argument("--run_dir", type=str, default=None, help="Diretório modelos_kf/<dataset>_<tmodel>")
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir) if args.run_dir else _discover_run_dir(None)
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    dataset, tmodel = _name_from_run_dir(run_dir)
    out_path = run_dir / f"{dataset}_{tmodel}_resultados.txt"

    metrics_by_fold, times_by_fold = _collect_all(run_dir)

    # ---- Agregar por fold (média nas seeds) ----
    folds_sorted = sorted(metrics_by_fold.keys())
    per_fold_means = []   # lista de dicts com médias por fold
    for fidx in folds_sorted:
        arr = metrics_by_fold[fidx]
        # Agrega por métrica
        accs  = [d.get("acc")  for d in arr if "acc" in d]
        precs = [d.get("prec") for d in arr if "prec" in d]
        recs  = [d.get("rec")  for d in arr if "rec" in d]
        f1s   = [d.get("f1")   for d in arr if "f1" in d]
        losses= [d.get("loss") for d in arr if "loss" in d]

        # média entre seeds (fold fixo)
        fold_mean = {
            "acc":  mean(accs)   if accs  else float("nan"),
            "prec": mean(precs)  if precs else float("nan"),
            "rec":  mean(recs)   if recs  else float("nan"),
            "f1":   mean(f1s)    if f1s   else float("nan"),
            "loss": mean(losses) if losses else float("nan"),
        }
        per_fold_means.append(fold_mean)

    # ---- Escrever arquivo de saída ----
    with out_path.open("w", encoding="utf-8") as f:
        for i, fold_mean in zip(folds_sorted, per_fold_means):
            f.write(f"Fold {i}:\n")
            f.write(f"  Acurácia: { _format_float(fold_mean['acc'], 6) }\n")
            f.write(f"  Precisão: { _format_float(fold_mean['prec'], 6) }\n")
            f.write(f"  Recall:   { _format_float(fold_mean['rec'], 6) }\n")
            f.write(f"  F1-score: { _format_float(fold_mean['f1'], 6) }\n\n")
            f.write(f"  Test Loss: { _format_float(fold_mean['loss'], 6) }\n\n")

        # ---- Resumo final (média & DP entre folds) ----
        accs_f  = [d["acc"]  for d in per_fold_means if not math.isnan(d["acc"])]
        precs_f = [d["prec"] for d in per_fold_means if not math.isnan(d["prec"])]
        recs_f  = [d["rec"]  for d in per_fold_means if not math.isnan(d["rec"])]
        f1s_f   = [d["f1"]   for d in per_fold_means if not math.isnan(d["f1"])]
        losses_f= [d["loss"] for d in per_fold_means if not math.isnan(d["loss"])]

        m_acc,  s_acc  = _agg_mean_std(accs_f)
        m_prec, s_prec = _agg_mean_std(precs_f)
        m_rec,  s_rec  = _agg_mean_std(recs_f)
        m_f1,   s_f1   = _agg_mean_std(f1s_f)
        m_loss, s_loss = _agg_mean_std(losses_f)

        f.write("=== Métricas Finais ===\n")
        f.write(f"Acurácia: Média={_format_float(m_acc, 6)}, Desvio={_format_float(s_acc, 6)}\n")
        f.write(f"Precisão: Média={_format_float(m_prec, 6)}, Desvio={_format_float(s_prec, 6)}\n")
        f.write(f"Recall:   Média={_format_float(m_rec, 6)}, Desvio={_format_float(s_rec, 6)}\n")
        f.write(f"F1-score: Média={_format_float(m_f1, 6)}, Desvio={_format_float(s_f1, 6)}\n")
        f.write(f"Test Loss: Média={_format_float(m_loss, 6)}, Desvio={_format_float(s_loss, 6)}\n\n")

        # ---- (Opcional) Resumo de tempos entre folds (média entre seeds por fold -> média/dp entre folds) ----
        if times_by_fold:
            f.write("=== Tempos (resumo) ===\n")
            def _agg_time(key: str) -> Tuple[float,float]:
                fold_means = []
                for fid, arr in sorted(times_by_fold.items()):
                    vals = [d.get(key) for d in arr if key in d]
                    if vals:
                        fold_means.append(mean(vals))
                return _agg_mean_std(fold_means)

            for label, key in [
                ("train_time_sec", "train_time_sec"),
                ("test_time_sec", "test_time_sec"),
                ("test_inf_ms_per_sample", "ms_per_sample"),
                ("throughput_samples_per_sec", "throughput"),
            ]:
                m, s = _agg_time(key)
                f.write(f"{label}: Média={_format_float(m, 6)}, Desvio={_format_float(s, 6)}\n")

    print(f"✓ Arquivo salvo: {out_path}")

if __name__ == "__main__":
    main()
