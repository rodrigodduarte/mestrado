#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
estatisticas_ne_plus.py — extrator robusto e completo para o layout seed/fold do train_kf_db.py

Mantém saída em TXT (<dataset>_<tmodel>_resultados.txt), porém com estatísticas ampliadas
para facilitar posterior tabulação e análise (sem depender de CSV/JSON).

Novidades em relação à versão anterior:
- Por fold: média e desvio-padrão entre seeds (antes só média).
- Por seed: agregação sobre folds (média/DP) e identificação da melhor seed (por acurácia).
- Entre folds: média/DP, erro-padrão e IC95% (normal).
- Tempos (train/test/inf/throughput) com o mesmo conjunto de estatísticas.
- Contagens e integridade: nº de folds detectados, seeds por fold, folds/arquivos ausentes.
- Snapshot de hiperparâmetros (se 'hyperparams_used.json' existir).
- Bloco final "JSON_SUMMARY:" com um dicionário compacto em uma única linha
  (permite parsing programático sem mudar o formato TXT predominante).

Uso:
  python estatisticas_ne_plus.py --run_dir modelos_kf/Flavia_convnext_t
  # ou:
  python estatisticas_ne_plus.py
"""

from __future__ import annotations
import json, math, re, sys
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
    patterns = {
        "acc":   [r"\bacc\b", r"accuracy"],
        "prec":  [r"\bprec", r"precision"],
        "rec":   [r"\brec(?!order)", r"recall"],
        "f1":    [r"\bf1\b", r"f1[_\- ]?score"],
        "loss":  [r"\bloss\b", r"test[_\- ]?loss", r"criterion"],
    }
    out = {}
    if isinstance(test_dict, dict):
        for name, pats in patterns.items():
            k = _first_key(test_dict, pats)
            if k is not None:
                try:
                    out[name] = float(test_dict[k])
                except Exception:
                    pass
    return out

def _extract_times(lines: List[str]) -> Dict[str, float]:
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
    try:
        s = txt_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None, None
    lines = [ln.strip() for ln in s.splitlines()]
    test_json = None
    for ln in lines:
        if ln.lower().startswith("test_metrics_json:"):
            _, js = ln.split(":", 1)
            js = js.strip()
            test_json = _safe_load_json(js)
            if test_json:
                break
    metrics = _extract_metrics(test_json or {})
    times = _extract_times(lines)
    return (metrics or None, times or None)

# ------------------------- Coleta & Descoberta -------------------------

def _discover_run_dir(run_dir: Optional[Path]) -> Path:
    if run_dir:
        return run_dir
    base = Path("modelos_kf")
    if not base.exists():
        raise FileNotFoundError("Não encontrei a pasta 'modelos_kf'. Passe --run_dir explicitamente.")
    candidates = []
    for d in base.iterdir():
        if not d.is_dir(): continue
        if "_" not in d.name: continue
        if any((d / s).is_dir() for s in d.glob("seed_*")):
            candidates.append(d)
    if len(candidates) == 0:
        raise FileNotFoundError("Nenhum diretório de execução encontrado em modelos_kf/*_*. Passe --run_dir.")
    if len(candidates) > 1:
        names = ", ".join(c.name for c in candidates[:6])
        raise RuntimeError(f"Mais de um diretório candidato encontrado ({names}). Passe --run_dir.")
    return candidates[0]

def _name_from_run_dir(run_dir: Path) -> Tuple[str, str]:
    name = run_dir.name
    if "_" not in name:
        return name, "model"
    parts = name.split("_", 1)
    return parts[0], parts[1]

def _collect_all(run_dir: Path):
    metrics_by_fold: Dict[int, List[Dict[str,float]]] = {}
    times_by_fold: Dict[int, List[Dict[str,float]]] = {}
    seeds_found: List[str] = []
    missing: List[str] = []

    seed_dirs = sorted([d for d in run_dir.glob("seed_*") if d.is_dir()],
                       key=lambda p: (len(p.name), p.name))
    if not seed_dirs:
        raise FileNotFoundError(f"Nenhuma pasta 'seed_*' encontrada em {run_dir}")

    for sd in seed_dirs:
        seeds_found.append(sd.name)
        fold_dirs = sorted(sd.glob("fold_*"))
        if not fold_dirs:
            missing.append(f"{sd.name}/(nenhum fold encontrado)")
        for fold_dir in fold_dirs:
            if not fold_dir.is_dir():
                continue
            m = re.search(r"fold_(\d+)", fold_dir.name)
            if not m:
                continue
            fold_idx = int(m.group(1))
            txt_path = fold_dir / "resultados_fold.txt"
            if not txt_path.exists():
                missing.append(f"{sd.name}/{fold_dir.name}/resultados_fold.txt")
                continue
            metrics, times = _read_resultados_fold(txt_path)
            if metrics:
                metrics_by_fold.setdefault(fold_idx, []).append(metrics)
            else:
                missing.append(f"{sd.name}/{fold_dir.name}/(métricas ausentes)")
            if times:
                times_by_fold.setdefault(fold_idx, []).append(times)
            else:
                missing.append(f"{sd.name}/{fold_dir.name}/(tempos ausentes)")

    if not metrics_by_fold:
        raise RuntimeError("Não foi possível extrair métricas de nenhum resultados_fold.txt")

    return metrics_by_fold, times_by_fold, seeds_found, missing

# ------------------------- Estatística -------------------------

def _agg_mean_std(items: List[float]) -> Tuple[float, float]:
    if not items:
        return float("nan"), float("nan")
    if len(items) == 1:
        return items[0], 0.0
    return (mean(items), pstdev(items))

def _stderr(sd: float, n: int) -> float:
    if n <= 0 or math.isnan(sd):
        return float("nan")
    return sd / math.sqrt(n)

def _ci95(mean_val: float, sd: float, n: int) -> Tuple[float,float]:
    if n <= 0 or math.isnan(sd) or math.isnan(mean_val):
        return (float("nan"), float("nan"))
    half = 1.96 * sd / math.sqrt(n)
    return (mean_val - half, mean_val + half)

def _format(x: float, nd: int = 6) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "nan"
    return f"{x:.{nd}f}"

# ------------------------- Principal -------------------------

def main(argv: Optional[List[str]] = None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Extrai e agrega métricas/tempos de seed/fold e gera TXT completo.")
    ap.add_argument("--run_dir", type=str, default=None, help="Diretório modelos_kf/<dataset>_<tmodel>")
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir) if args.run_dir else _discover_run_dir(None)
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    dataset, tmodel = _name_from_run_dir(run_dir)
    out_path = run_dir / f"{dataset}_{tmodel}_resultados.txt"

    metrics_by_fold, times_by_fold, seeds_found, missing = _collect_all(run_dir)

    folds_sorted = sorted(metrics_by_fold.keys())

    # --- Por fold: média e DP entre seeds
    per_fold_stats = {}
    for fidx in folds_sorted:
        arr = metrics_by_fold[fidx]
        accs   = [d.get("acc")  for d in arr if "acc"  in d]
        precs  = [d.get("prec") for d in arr if "prec" in d]
        recs   = [d.get("rec")  for d in arr if "rec"  in d]
        f1s    = [d.get("f1")   for d in arr if "f1"   in d]
        losses = [d.get("loss") for d in arr if "loss" in d]

        per_fold_stats[fidx] = {
            "acc":   {"mean": mean(accs)   if accs   else float("nan"),
                      "std":  pstdev(accs)  if len(accs)  > 1 else 0.0, "n": len(accs)},
            "prec":  {"mean": mean(precs)  if precs  else float("nan"),
                      "std":  pstdev(precs) if len(precs) > 1 else 0.0, "n": len(precs)},
            "rec":   {"mean": mean(recs)   if recs   else float("nan"),
                      "std":  pstdev(recs)  if len(recs)  > 1 else 0.0, "n": len(recs)},
            "f1":    {"mean": mean(f1s)    if f1s    else float("nan"),
                      "std":  pstdev(f1s)   if len(f1s)   > 1 else 0.0, "n": len(f1s)},
            "loss":  {"mean": mean(losses) if losses else float("nan"),
                      "std":  pstdev(losses)if len(losses)> 1 else 0.0, "n": len(losses)},
        }

    # --- Entre folds: média/DP/SE/IC95 das médias por fold
    def _collect_fold_means(metric):
        vals = [per_fold_stats[f][metric]["mean"] for f in folds_sorted
                if not math.isnan(per_fold_stats[f][metric]["mean"])]
        return vals

    summary = {}
    for m in ["acc", "prec", "rec", "f1", "loss"]:
        vals = _collect_fold_means(m)
        m_mean, m_sd = _agg_mean_std(vals)
        se = _stderr(m_sd, len(vals))
        lo, hi = _ci95(m_mean, m_sd, len(vals))
        summary[m] = {"mean": m_mean, "std": m_sd, "se": se, "ci95": (lo, hi), "n_folds": len(vals)}

    # --- Tempos: por fold (média/DP entre seeds) e resumo entre folds
    per_fold_time = {}
    for fidx, arr in sorted(times_by_fold.items()):
        for key in ["train_time_sec","test_time_sec","ms_per_sample","throughput"]:
            vals = [d.get(key) for d in arr if key in d]
            if not vals:
                continue
            per_fold_time.setdefault(fidx, {})[key] = {
                "mean": mean(vals),
                "std": pstdev(vals) if len(vals) > 1 else 0.0,
                "n": len(vals)
            }
    time_summary = {}
    for key in ["train_time_sec","test_time_sec","ms_per_sample","throughput"]:
        fold_means = [per_fold_time[f][key]["mean"] for f in sorted(per_fold_time.keys())
                      if key in per_fold_time[f]]
        if fold_means:
            m_mean, m_sd = _agg_mean_std(fold_means)
            se = _stderr(m_sd, len(fold_means))
            lo, hi = _ci95(m_mean, m_sd, len(fold_means))
            time_summary[key] = {"mean": m_mean, "std": m_sd, "se": se, "ci95": (lo,hi), "n_folds": len(fold_means)}

    # --- Por seed: agregação sobre folds (média/DP por seed)
    # Para isso, precisamos revarrer para montar (seed -> lista por fold)
    seed_fold_metrics: Dict[str, Dict[str, List[float]]] = {}
    seed_dirs = sorted([d for d in run_dir.glob("seed_*") if d.is_dir()],
                       key=lambda p: (len(p.name), p.name))
    for sd in seed_dirs:
        seed_name = sd.name
        for fold_dir in sorted(sd.glob("fold_*")):
            txt_path = fold_dir / "resultados_fold.txt"
            if not txt_path.exists():
                continue
            metrics, _ = _read_resultados_fold(txt_path)
            if not metrics:
                continue
            for key in ["acc","prec","rec","f1","loss"]:
                if key in metrics:
                    seed_fold_metrics.setdefault(seed_name, {}).setdefault(key, []).append(metrics[key])

    seed_summary = {}
    for seed_name, mdict in seed_fold_metrics.items():
        seed_summary[seed_name] = {}
        for key, vals in mdict.items():
            m_mean, m_sd = _agg_mean_std(vals)
            seed_summary[seed_name][key] = {"mean": m_mean, "std": m_sd, "n_folds": len(vals)}

    # melhor seed por acurácia média
    best_seed = None
    best_acc = -1.0
    for seed_name, mdict in seed_summary.items():
        acc_mean = mdict.get("acc", {}).get("mean", float("nan"))
        if not math.isnan(acc_mean) and acc_mean > best_acc:
            best_acc = acc_mean
            best_seed = seed_name

    # --- Hiperparâmetros (snapshot se existir)
    hparams = None
    hp_path = run_dir / "hyperparams_used.json"
    if hp_path.exists():
        try:
            hparams = json.loads(hp_path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            hparams = None

    # ------------------------- Saída TXT -------------------------
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"# Resultados consolidados — {dataset}_{tmodel}\n\n")

        # Info geral
        f.write("=== Informações do Experimento ===\n")
        f.write(f"Run dir: {run_dir}\n")
        f.write(f"Seeds detectadas: {', '.join(seeds_found) if seeds_found else 'nenhuma'}\n")
        f.write(f"Folds detectados: {', '.join(str(x) for x in folds_sorted) if folds_sorted else 'nenhum'}\n")
        if missing:
            f.write("Avisos (arquivos/itens ausentes):\n")
            for m in missing[:50]:
                f.write(f"  - {m}\n")
            if len(missing) > 50:
                f.write(f"  ... (+{len(missing)-50} itens)\n")
        f.write("\n")

        # Hiperparâmetros
        if hparams:
            f.write("=== Hiperparâmetros (snapshot) ===\n")
            try:
                # imprimir alguns mais relevantes, sem estourar o TXT
                keys_show = ["tmodel","name_dataset","epochs","learning_rate","batch_size",
                             "drop_path_rate","num_classes","label_smoothing","optimizer_momentum",
                             "weight_decay","layer_scale","image_size"]
                for k in keys_show:
                    if k in hparams:
                        f.write(f"{k}: {hparams[k]}\n")
            except Exception:
                pass
            f.write("\n")

        # Por fold
        for fidx in folds_sorted:
            f.write(f"== Fold {fidx} ==\n")
            stats = per_fold_stats[fidx]
            for key, label in [("acc","Acurácia"),("prec","Precisão"),
                               ("rec","Recall"),("f1","F1-score"),("loss","Test Loss")]:
                dd = stats[key]
                f.write(f"{label}: mean={_format(dd['mean'])} std={_format(dd['std'])} n_seeds={dd['n']}\n")
            if fidx in per_fold_time:
                f.write("Tempos:\n")
                tt = per_fold_time[fidx]
                def line(k, lab):
                    if k in tt:
                        d = tt[k]
                        f.write(f"  {lab}: mean={_format(d['mean'])} std={_format(d['std'])} n_seeds={d['n']}\n")
                line("train_time_sec","train_time_sec")
                line("test_time_sec","test_time_sec")
                line("ms_per_sample","test_inf_ms_per_sample")
                line("throughput","throughput_samples_per_sec")
            f.write("\n")

        # Resumo entre folds
        f.write("=== Métricas — Resumo entre folds ===\n")
        for key, label in [("acc","Acurácia"),("prec","Precisão"),("rec","Recall"),("f1","F1-score"),("loss","Test Loss")]:
            dd = summary[key]
            ci = dd["ci95"]
            f.write(f"{label}: mean={_format(dd['mean'])} std={_format(dd['std'])} se={_format(dd['se'])} "
                    f"ci95=({_format(ci[0])},{_format(ci[1])}) n_folds={dd['n_folds']}\n")
        f.write("\n")

        if time_summary:
            f.write("=== Tempos — Resumo entre folds ===\n")
            labels = [("train_time_sec","train_time_sec"),
                      ("test_time_sec","test_time_sec"),
                      ("ms_per_sample","test_inf_ms_per_sample"),
                      ("throughput","throughput_samples_per_sec")]
            for key, lab in labels:
                if key in time_summary:
                    dd = time_summary[key]
                    ci = dd["ci95"]
                    f.write(f"{lab}: mean={_format(dd['mean'])} std={_format(dd['std'])} se={_format(dd['se'])} "
                            f"ci95=({_format(ci[0])},{_format(ci[1])}) n_folds={dd['n_folds']}\n")
            f.write("\n")

        # Por seed (agregado sobre folds)
        if seed_summary:
            f.write("=== Resumo por Seed (agregado sobre folds) ===\n")
            f.write(f"Melhor seed por acurácia média: {best_seed} (acc_mean={_format(best_acc)})\n")
            for seed_name in sorted(seed_summary.keys(), key=lambda s: (len(s), s)):
                f.write(f"- {seed_name}:\n")
                for key, label in [("acc","Acurácia"),("prec","Precisão"),("rec","Recall"),("f1","F1-score"),("loss","Test Loss")]:
                    if key in seed_summary[seed_name]:
                        d = seed_summary[seed_name][key]
                        f.write(f"  {label}: mean={_format(d['mean'])} std={_format(d['std'])} n_folds={d['n_folds']}\n")
            f.write("\n")

        # Bloco JSON compacto para parsing futuro (opcional)
        compact = {
            "dataset": dataset, "tmodel": tmodel,
            "folds": folds_sorted, "seeds": seeds_found,
            "per_fold_stats": per_fold_stats, "summary": summary,
            "per_fold_time": per_fold_time, "time_summary": time_summary,
            "seed_summary": seed_summary, "best_seed": best_seed, "best_seed_acc_mean": best_acc
        }
        try:
            f.write("JSON_SUMMARY: ")
            f.write(json.dumps(compact, ensure_ascii=False, separators=(",",":")))
            f.write("\n")
        except Exception:
            pass

    print(f"✓ Arquivo salvo: {out_path}")

if __name__ == "__main__":
    main()
