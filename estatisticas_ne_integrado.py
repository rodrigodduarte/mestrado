#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
estatisticas_ne_integrado.py — usa config.py para descobrir o dataset e agrega resultados (seed/fold).

Principais recursos:
- Lê o arquivo de config (Python) via runpy e extrai TRAIN_DIR → deduz o nome do dataset.
- Descobre a pasta "modelos_kf" a partir do diretório do config ou do CWD (subindo e/ou varredura recursiva).
- Encontra TODOS os runs em modelos_kf cujo nome começa com "<dataset>_" e que contenham "seed_*".
- Para cada run encontrado, gera <dataset>_<tmodel>_resultados.txt com estatísticas completas:
  * Por fold: média e DP entre seeds (acc, prec, rec, f1, loss) + tempos.
  * Resumo entre folds: média/DP/SE/IC95.
  * Por seed: média/DP via folds + melhor seed por acc.
  * Bloco final JSON_SUMMARY para parsing programático.
- Robusto a variações de chave nas métricas dentro de test_metrics_json.
- Aceita override de --run_dir se quiser processar só um run específico.

Uso:
  python estatisticas_ne_integrado.py --config /caminho/para/config.py
  # ou:
  python estatisticas_ne_integrado.py  (ele tentará achar um config.py próximo)
"""

from __future__ import annotations
import argparse, json, math, re, runpy, sys
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
    for pat in patterns:
        rx = re.compile(pat, flags=re.IGNORECASE)
        for k in d.keys():
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
            test_json = _safe_load_json(js.strip())
            if test_json:
                break
    metrics = _extract_metrics(test_json or {})
    times = _extract_times(lines)
    return (metrics or None, times or None)

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

def _agg_mean_std(arr: List[float]) -> Tuple[float,float]:
    if not arr:
        return float("nan"), float("nan")
    if len(arr) == 1:
        return arr[0], 0.0
    return mean(arr), pstdev(arr)

# ------------------------- Descoberta baseada no config -------------------------

def _load_config_dict(cfg_path: Optional[Path]) -> Tuple[Optional[dict], Optional[Path]]:
    """Carrega config.py via runpy. Retorna (dict, base_dir_do_config_ou_None)."""
    search_roots = []
    if cfg_path:
        p = Path(cfg_path).resolve()
        if p.is_file():
            try:
                d = runpy.run_path(str(p))
                return d, p.parent
            except Exception:
                pass
        # se apontou para diretório, tente achar config.py dentro
        if p.is_dir():
            cand = p / "config.py"
            if cand.exists():
                try:
                    d = runpy.run_path(str(cand))
                    return d, p
                except Exception:
                    pass
    # tentar localizar config.py subindo desde CWD
    cur = Path.cwd().resolve()
    for _ in range(8):
        cand = cur / "config.py"
        if cand.exists():
            try:
                d = runpy.run_path(str(cand))
                return d, cur
            except Exception:
                pass
        if cur.parent == cur:
            break
        cur = cur.parent

    # busca recursiva superficial a partir do CWD
    for cand in Path.cwd().resolve().rglob("config.py"):
        try:
            d = runpy.run_path(str(cand))
            return d, cand.parent
        except Exception:
            continue
    return None, None

def _dataset_from_config(cfg: dict) -> Optional[str]:
    td = cfg.get("TRAIN_DIR") or cfg.get("train_dir") or cfg.get("TRAIN_DATA_DIR")
    if not td:
        return None
    p = Path(td)
    # se termina com /train, dataset é o pai
    if p.name.lower() == "train":
        return p.parent.name
    # caso contrário, use o último nome
    return p.name

def _find_modelos_kf(start_dirs: List[Path]) -> Optional[Path]:
    # tentar direto
    for base in start_dirs + [Path.cwd().resolve()]:
        cand = base / "modelos_kf"
        if cand.is_dir():
            return cand
        # subir alguns níveis
        cur = base
        for _ in range(6):
            cand = cur / "modelos_kf"
            if cand.is_dir():
                return cand
            if cur.parent == cur:
                break
            cur = cur.parent
    # varredura recursiva limitada
    for base in start_dirs + [Path.cwd().resolve()]:
        hits = list(base.rglob("modelos_kf"))
        if hits:
            return hits[0]
    # fallback: busca global a partir do CWD
    for cand in Path.cwd().resolve().rglob("modelos_kf"):
        return cand
    return None

def _list_runs_for_dataset(mkf: Path, dataset: str) -> List[Path]:
    runs = []
    # diretos: modelos_kf/<dataset>_*
    for d in mkf.iterdir():
        if d.is_dir() and d.name.startswith(f"{dataset}_"):
            if any((d / s).is_dir() for s in d.glob("seed_*")):
                runs.append(d)
    # se nada achou, varrer recursivo (caso de estrutura aninhada)
    if not runs:
        for d in mkf.rglob(f"{dataset}_*"):
            if d.is_dir() and any((d / s).is_dir() for s in d.glob("seed_*")):
                runs.append(d)
    return sorted(set(runs), key=lambda p: p.name)

def _name_from_run_dir(run_dir: Path) -> Tuple[str, str]:
    name = run_dir.name
    if "_" not in name:
        return name, "model"
    parts = name.split("_", 1)
    return parts[0], parts[1]

# ------------------------- Coleta e Agregação -------------------------

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

def _write_report_for_run(run_dir: Path) -> Path:
    dataset, tmodel = _name_from_run_dir(run_dir)
    out_path = run_dir / f"{dataset}_{tmodel}_resultados.txt"

    metrics_by_fold, times_by_fold, seeds_found, missing = _collect_all(run_dir)
    folds_sorted = sorted(metrics_by_fold.keys())

    # Por fold: mean/std entre seeds
    per_fold_stats = {}
    for fidx in folds_sorted:
        arr = metrics_by_fold[fidx]
        accs   = [d.get("acc")  for d in arr if "acc"  in d]
        precs  = [d.get("prec") for d in arr if "prec" in d]
        recs   = [d.get("rec")  for d in arr if "rec"  in d]
        f1s    = [d.get("f1")   for d in arr if "f1"   in d]
        losses = [d.get("loss") for d in arr if "loss" in d]
        per_fold_stats[fidx] = {
            "acc":  {"mean": mean(accs)   if accs   else float("nan"), "std": pstdev(accs)   if len(accs)  > 1 else 0.0, "n": len(accs)},
            "prec": {"mean": mean(precs)  if precs  else float("nan"), "std": pstdev(precs)  if len(precs) > 1 else 0.0, "n": len(precs)},
            "rec":  {"mean": mean(recs)   if recs   else float("nan"), "std": pstdev(recs)   if len(recs)  > 1 else 0.0, "n": len(recs)},
            "f1":   {"mean": mean(f1s)    if f1s    else float("nan"), "std": pstdev(f1s)    if len(f1s)   > 1 else 0.0, "n": len(f1s)},
            "loss": {"mean": mean(losses) if losses else float("nan"), "std": pstdev(losses) if len(losses)> 1 else 0.0, "n": len(losses)},
        }

    # Entre folds: mean/std/se/ci95
    def _collect_fold_means(metric):
        return [per_fold_stats[f][metric]["mean"] for f in folds_sorted
                if not math.isnan(per_fold_stats[f][metric]["mean"])]
    summary = {}
    for m in ["acc","prec","rec","f1","loss"]:
        vals = _collect_fold_means(m)
        m_mean, m_sd = _agg_mean_std(vals)
        se = _stderr(m_sd, len(vals))
        lo, hi = _ci95(m_mean, m_sd, len(vals))
        summary[m] = {"mean": m_mean, "std": m_sd, "se": se, "ci95": (lo, hi), "n_folds": len(vals)}

    # Tempos
    per_fold_time = {}
    for fidx, arr in sorted(times_by_fold.items()):
        for key in ["train_time_sec","test_time_sec","ms_per_sample","throughput"]:
            vals = [d.get(key) for d in arr if key in d]
            if not vals: continue
            per_fold_time.setdefault(fidx, {})[key] = {
                "mean": mean(vals), "std": pstdev(vals) if len(vals) > 1 else 0.0, "n": len(vals)
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

    # Por seed: agregação sobre folds
    seed_fold_metrics: Dict[str, Dict[str, List[float]]] = {}
    for sd in sorted([d for d in run_dir.glob("seed_*") if d.is_dir()],
                     key=lambda p: (len(p.name), p.name)):
        seed_name = sd.name
        for fold_dir in sorted(sd.glob("fold_*")):
            txt_path = fold_dir / "resultados_fold.txt"
            if not txt_path.exists(): continue
            metrics, _ = _read_resultados_fold(txt_path)
            if not metrics: continue
            for key in ["acc","prec","rec","f1","loss"]:
                if key in metrics:
                    seed_fold_metrics.setdefault(seed_name, {}).setdefault(key, []).append(metrics[key])

    seed_summary = {}
    best_seed, best_acc = None, -1.0
    for seed_name, mdict in seed_fold_metrics.items():
        seed_summary[seed_name] = {}
        for key, vals in mdict.items():
            m_mean, m_sd = _agg_mean_std(vals)
            seed_summary[seed_name][key] = {"mean": m_mean, "std": m_sd, "n_folds": len(vals)}
        acc_m = seed_summary[seed_name].get("acc", {}).get("mean", float("nan"))
        if not math.isnan(acc_m) and acc_m > best_acc:
            best_acc, best_seed = acc_m, seed_name

    # Hiperparâmetros (opcional)
    hparams = None
    hp_path = run_dir / "hyperparams_used.json"
    if hp_path.exists():
        try:
            hparams = json.loads(hp_path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            hparams = None

    # Escreve
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"# Resultados consolidados — {dataset}_{tmodel}\n\n")
        f.write("=== Informações do Experimento ===\n")
        f.write(f"Run dir: {run_dir}\n")
        f.write(f"Seeds detectadas: {', '.join(sorted(seeds_found, key=lambda s: (len(s), s)))}\n")
        f.write(f"Folds detectados: {', '.join(str(x) for x in sorted(folds_sorted))}\n")
        if missing:
            f.write("Avisos (arquivos/itens ausentes):\n")
            for m in missing[:50]:
                f.write(f"  - {m}\n")
            if len(missing) > 50:
                f.write(f"  ... (+{len(missing)-50} itens)\n")
        f.write("\n")

        if hparams:
            f.write("=== Hiperparâmetros (snapshot) ===\n")
            keys_show = ["tmodel","name_dataset","epochs","learning_rate","batch_size",
                         "drop_path_rate","num_classes","label_smoothing","optimizer_momentum",
                         "weight_decay","layer_scale","image_size"]
            for k in keys_show:
                if k in hparams:
                    f.write(f"{k}: {hparams[k]}\n")
            f.write("\n")

        for fidx in sorted(folds_sorted):
            f.write(f"== Fold {fidx} ==\n")
            stats = per_fold_stats[fidx]
            for key, label in [("acc","Acurácia"),("prec","Precisão"),("rec","Recall"),("f1","F1-score"),("loss","Test Loss")]:
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

        f.write("=== Métricas — Resumo entre folds ===\n")
        for key, label in [("acc","Acurácia"),("prec","Precisão"),("rec","Recall"),("f1","F1-score"),("loss","Test Loss")]:
            dd = summary[key]
            lo, hi = dd["ci95"]
            f.write(f"{label}: mean={_format(dd['mean'])} std={_format(dd['std'])} se={_format(dd['se'])} "
                    f"ci95=({_format(lo)},{_format(hi)}) n_folds={dd['n_folds']}\n")
        f.write("\n")

        if time_summary:
            f.write("=== Tempos — Resumo entre folds ===\n")
            labels = [("train_time_sec","train_time_sec"),
                      ("test_time_sec","test_time_sec"),
                      ("ms_per_sample","test_inf_ms_per_sample"),
                      ("throughput","throughput_samples_per_sec")]
            for key, lab in labels:
                if key in time_summary:
                    dd = time_summary[key]; lo, hi = dd["ci95"]
                    f.write(f"{lab}: mean={_format(dd['mean'])} std={_format(dd['std'])} se={_format(dd['se'])} "
                            f"ci95=({_format(lo)},{_format(hi)}) n_folds={dd['n_folds']}\n")
            f.write("\n")

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

        compact = {
            "dataset": dataset, "tmodel": tmodel,
            "folds": sorted(folds_sorted), "seeds": sorted(seeds_found, key=lambda s: (len(s), s)),
            "per_fold_stats": per_fold_stats, "summary": summary,
            "per_fold_time": per_fold_time, "time_summary": time_summary,
            "seed_summary": seed_summary, "best_seed": best_seed, "best_seed_acc_mean": best_acc
        }
        f.write("JSON_SUMMARY: ")
        f.write(json.dumps(compact, ensure_ascii=False, separators=(",",":")))
        f.write("\n")

    return out_path

def _main(argv=None):
    ap = argparse.ArgumentParser(description="Extrai estatísticas usando config.py para detectar dataset.")
    ap.add_argument("--config", type=str, default=None, help="Caminho para config.py (opcional).")
    ap.add_argument("--run_dir", type=str, default=None, help="Processar apenas este run (sobrepõe detecção por dataset).")
    args = ap.parse_args(argv)

    cfg, cfg_dir = _load_config_dict(Path(args.config) if args.config else None)
    dataset = _dataset_from_config(cfg) if cfg else None

    # Se --run_dir foi passado, processa só ele
    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
        if not run_dir.exists():
            raise FileNotFoundError(f"--run_dir não existe: {run_dir}")
        out = _write_report_for_run(run_dir)
        print(f"✓ Arquivo salvo: {out}")
        return

    # Caso contrário, usamos dataset do config para achar runs
    if not dataset:
        raise RuntimeError("Não foi possível obter o dataset a partir do config (precisa de TRAIN_DIR). Passe --run_dir.")

    # Descobrir modelos_kf
    mkf = _find_modelos_kf([d for d in [cfg_dir] if d] or [])
    if not mkf:
        raise FileNotFoundError("Não encontrei a pasta 'modelos_kf' próximo ao config nem no CWD. Passe --run_dir.")

    runs = _list_runs_for_dataset(mkf, dataset)
    if not runs:
        raise FileNotFoundError(f"Nenhum run encontrado em {mkf} com prefixo '{dataset}_' contendo seeds. Passe --run_dir.")

    saved = []
    for rd in runs:
        try:
            out = _write_report_for_run(rd)
            saved.append(out)
        except Exception as e:
            print(f"[!] Falha ao processar {rd}: {e}", file=sys.stderr)

    if saved:
        print("✓ Arquivos gerados:")
        for p in saved:
            print("  -", p)
    else:
        raise RuntimeError("Nenhum run foi processado com sucesso.")

if __name__ == "__main__":
    _main()
