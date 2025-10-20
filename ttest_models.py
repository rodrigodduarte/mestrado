#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paired t-test e relatório unificado em .txt para dois modelos.
Lê seed_*/fold_*/resultados_fold.txt de cada pasta, usa tmodel de hyperparams_used.json
(quando existir) para nomear os modelos, e imprime TUDO no stdout (ou em --out).
Conteúdo impresso:
  - Testes t pareados (fold-wise e seed-wise) para a métrica de teste escolhida.
  - Tabelas FoldxSeed para a métrica escolhida (uma por modelo).
  - Tabelas FoldxSeed para TODAS as métricas de teste (test.*) e validação (val.*).
  - Tabelas FoldxSeed para eficiência/recursos (train/test time, ms/sample, throughput,
    memória, tamanho do checkpoint, best_epoch, e contagens se existirem).
  - Tabelas FoldxSeed de DIFERENÇA pareada A-B para todo campo numérico em comum.
  - (Opcional) Dump de hyperparams_used.json de cada modelo.

Exemplo:
python ttest_models.py \  --modelA ~/Documentos/projeto/modelos_kf/soja_convnext_t \  --modelB ~/Documentos/projeto/modelos_kf/soja_vector \  --metric accuracy \  --out ~/Documentos/projeto/ttest_soja_accuracy.txt
"""
import argparse, os, json, math, sys, re
from pathlib import Path
from collections import defaultdict
import numpy as np

try:
    from scipy import stats
except Exception as e:
    raise SystemExit("scipy is required. Try: pip install scipy") from e

CAND_KEYS = [
    "test_accuracy","accuracy","acc",
    "test_f1","f1","f1_score",
    "test_precision","precision","prec",
    "test_recall","recall","rec",
    "test_loss","loss"
]

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
    def flush(self):
        for f in self.files:
            try:
                f.flush()
            except Exception:
                pass

_NUM_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$")

def _to_number_or_str(x):
    if isinstance(x, (int, float)):
        return float(x)
    xs = str(x).strip()
    if xs.lower() in ("true","false"):
        return xs.lower() == "true"
    if xs.lower() in ("nan","inf","-inf","+inf"):
        return float("nan")
    if _NUM_RE.match(xs):
        try:
            return float(xs)
        except Exception:
            return xs
    return xs

def _load_json_relaxed(jstr: str):
    try:
        return json.loads(jstr)
    except Exception:
        pass
    j2 = jstr.replace("'", '"')
    j2 = re.sub(r"\bNaN\b", "null", j2, flags=re.IGNORECASE)
    try:
        return json.loads(j2)
    except Exception:
        return {}

def parse_result_file(path: Path):
    data = {"val_metrics_json": {}, "test_metrics_json": {}}
    seed = fold = None
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or ":" not in line:
                continue
            k, _, v = line.partition(":")
            key = k.strip()
            val = v.strip()
            if key == "Seed":
                try: seed = int(val)
                except: seed = _to_number_or_str(val)
                data["Seed"] = seed
            elif key == "Fold":
                try: fold = int(val)
                except: fold = _to_number_or_str(val)
                data["Fold"] = fold
            elif key in ("val_metrics_json", "test_metrics_json"):
                data[key] = _load_json_relaxed(val)
            else:
                data[key] = _to_number_or_str(val)
    return seed, fold, data

def gather_results(root: Path):
    out = {}
    for p in root.rglob("resultados_fold.txt"):
        s,f,dat = parse_result_file(p)
        if s is None or f is None:
            continue
        out[(s,f)] = dat
    return out

def infer_model_label(root: Path, manual_label: str|None = None) -> str:
    if manual_label:
        return manual_label
    hp_candidates = []
    hp_root = root / "hyperparams_used.json"
    if hp_root.exists():
        hp_candidates.append(hp_root)
    if not hp_candidates:
        for p in root.glob("**/hyperparams_used.json"):
            hp_candidates.append(p); break
    for hp in hp_candidates:
        try:
            data = json.loads(hp.read_text(encoding="utf-8"))
            for key in ("tmodel", "TMODEL", "model", "name_model"):
                if key in data and isinstance(data[key], str) and data[key].strip():
                    return data[key].strip()
        except Exception:
            pass
    return root.name

def pick_metric_key(test_metrics: dict, prefer: str|None):
    if prefer and prefer in test_metrics:
        return prefer
    for k in CAND_KEYS:
        if k in test_metrics and isinstance(test_metrics[k], (int,float)):
            return k
    for k,v in test_metrics.items():
        if isinstance(v, (int,float)):
            return k
    raise ValueError("Não foi possível identificar uma métrica numérica em test_metrics_json. "
                     f"Chaves vistas: {list(test_metrics.keys())[:10]}")

def ci95_of_mean(x: np.ndarray):
    n = len(x)
    mean = float(np.mean(x)) if n else float("nan")
    sd = float(np.std(x, ddof=1)) if n>1 else 0.0
    if n>1:
        tcrit = stats.t.ppf(0.975, df=n-1)
        half = tcrit*sd/math.sqrt(n)
        return (mean-half, mean+half), mean, sd
    else:
        return (mean, mean), mean, sd

def paired_ttest(a: np.ndarray, b: np.ndarray):
    if len(a) != len(b):
        raise ValueError("Vetores a e b com tamanhos diferentes.")
    d = a - b
    n = len(d)
    if n < 2:
        return {"n": n, "mean_diff": float(np.mean(d) if n else float('nan')),
                "sd_diff": float('nan'), "t": float('nan'),
                "p_two": float('nan'), "cohen_d": float('nan'),
                "ci95_diff": (float('nan'), float('nan'))}
    mean = float(np.mean(d))
    sd = float(np.std(d, ddof=1))
    se = sd/math.sqrt(n)
    t_stat = mean / se if se>0 else float('inf')
    p_two = float(stats.t.sf(abs(t_stat), df=n-1) * 2)
    d_cohen = mean/sd if sd>0 else float('inf')
    tcrit = stats.t.ppf(0.975, df=n-1)
    ci = (mean - tcrit*se, mean + tcrit*se)
    return {"n": n, "mean_diff": mean, "sd_diff": sd, "t": t_stat, "p_two": p_two,
            "cohen_d": d_cohen, "ci95_diff": ci}

def build_matrix(map_sf_val: dict, seeds: list[int], folds: list[int]):
    M = np.full((len(folds), len(seeds)), np.nan)
    for i, f in enumerate(folds):
        for j, s in enumerate(seeds):
            v = map_sf_val.get((s, f), np.nan)
            if isinstance(v, (int,float)):
                M[i, j] = float(v)
    return M

def fmt_num(x):
    return "NA".rjust(8) if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{x:8.4f}"

def print_matrix_table(title: str, seeds: list[int], folds: list[int], M: np.ndarray):
    header = ["fold\\seed"] + [f"s{int(s)}" for s in seeds] + ["mean_fold"]
    widths = [max(9, len(h)) for h in header]
    def print_row(cells):
        print(" | ".join(str(c).rjust(w) for c, w in zip(cells, widths)))
    print(title)
    print_row(header)
    print("-" * (sum(widths) + 3*(len(widths)-1)))
    row_means = np.nanmean(M, axis=1) if M.size else np.array([])
    for i, f in enumerate(folds):
        cells = [f"fold_{f}"] + [fmt_num(M[i, j]) for j in range(len(seeds))] + [fmt_num(row_means[i] if len(row_means)>i else float("nan"))]
        print_row(cells)
    col_means = np.nanmean(M, axis=0) if M.size else np.array([])
    agg = float(np.nanmean(M)) if M.size else float("nan")
    print("-" * (sum(widths) + 3*(len(widths)-1)))
    tail = ["mean_seed"] + [fmt_num(col_means[j] if len(col_means)>j else float("nan")) for j in range(len(seeds))] + [fmt_num(agg)]
    print_row(tail)
    print()

NUMERIC_PREF_KEYS = {
    "best_epoch",
    "train_time_sec","test_time_sec","test_inf_ms_per_sample","throughput_samples_per_sec",
    "max_gpu_mem_mb","best_checkpoint_size_mb",
    "n_train","n_val","n_test","num_params"
}
CATEG_KEYS = {"balance_mode","class_weights_injected"}

def flatten_record(data: dict):
    flat = {}
    for k, v in data.items():
        if k in ("val_metrics_json","test_metrics_json","Seed","Fold"):
            continue
        if k in NUMERIC_PREF_KEYS:
            flat[k] = _to_number_or_str(v)
        elif k in CATEG_KEYS:
            flat[k] = v
    for pref, key in (("val","val_metrics_json"), ("test","test_metrics_json")):
        md = data.get(key, {}) or {}
        for mk, mv in md.items():
            val = _to_number_or_str(mv)
            flat[f"{pref}.{mk}"] = val
    return flat

def collect_all_numeric_fields(results: dict):
    keys = set()
    for dat in results.values():
        flat = flatten_record(dat)
        for k, v in flat.items():
            if isinstance(v, (int,float)) and not (isinstance(v,float) and np.isnan(v)):
                keys.add(k)
    return sorted(keys)

def main():
    ap = argparse.ArgumentParser(description="Paired t-test + unified report into a single .txt")
    ap.add_argument("--modelA", required=True, help="Pasta do modelo A (ex.: modelos_kf/soja_convnext_t)")
    ap.add_argument("--modelB", required=True, help="Pasta do modelo B (ex.: modelos_kf/soja_vector)")
    ap.add_argument("--metric", default=None, help="Chave de métrica de teste (ex.: accuracy, f1, test_loss). Se omitido, autodetecta.")
    ap.add_argument("--labelA", default=None, help="Rótulo manual para A (sobrepõe 'tmodel')")
    ap.add_argument("--labelB", default=None, help="Rótulo manual para B (sobrepõe 'tmodel')")
    ap.add_argument("--out", default=None, help="Se definido, salva TODO o relatório em um único .txt (sem novos diretórios).")
    ap.add_argument("--print_hparams", action="store_true", help="Se definido, imprime o hyperparams_used.json de cada modelo no final.")
    args = ap.parse_args()

    stdout_backup = sys.stdout
    out_fh = None
    if args.out:
        out_path = Path(os.path.expanduser(args.out))
        out_fh = out_path.open("w", encoding="utf-8")
        sys.stdout = Tee(sys.stdout, out_fh)

    rootA = Path(os.path.expanduser(args.modelA)).resolve()
    rootB = Path(os.path.expanduser(args.modelB)).resolve()
    if not rootA.exists() or not rootB.exists():
        raise SystemExit(f"Caminhos inválidos:\n A={rootA}\n B={rootB}")

    labelA = infer_model_label(rootA, args.labelA)
    labelB = infer_model_label(rootB, args.labelB)

    A = gather_results(rootA)
    B = gather_results(rootB)
    common = sorted(set(A).intersection(B))
    if not common:
        raise SystemExit("Não há pares (seed, fold) em comum entre as duas pastas. "
                         "Verifique se os dois modelos foram treinados com as mesmas seeds e folds.")

    ref_test_metrics = A[common[0]].get("test_metrics_json", {}) or {}
    mk = args.metric or pick_metric_key(ref_test_metrics, args.metric)
    if mk not in (B[common[0]].get("test_metrics_json", {}) or {}):
        keysA = set((A[common[0]].get("test_metrics_json", {}) or {}).keys())
        keysB = set((B[common[0]].get("test_metrics_json", {}) or {}).keys())
        inter = [k for k in CAND_KEYS if k in keysA and k in keysB]
        if inter:
            mk = inter[0]
        else:
            raise SystemExit(f"A métrica '{mk}' não está presente em ambos.\n"
                             f"Exemplos A={list(keysA)[:6]} | B={list(keysB)[:6]}")

    a_vals, b_vals = [], []
    for k in common:
        va = (A[k].get("test_metrics_json") or {}).get(mk, None)
        vb = (B[k].get("test_metrics_json") or {}).get(mk, None)
        if isinstance(va, (int,float)) and isinstance(vb, (int,float)):
            a_vals.append(float(va))
            b_vals.append(float(vb))
    a = np.array(a_vals, dtype=float)
    b = np.array(b_vals, dtype=float)

    print("# Unified Paired Report")
    print(f"Model A: {labelA} | Root: {rootA}")
    print(f"Model B: {labelB} | Root: {rootB}")
    print(f"Paired on (seed, fold). Using test metric: {mk}")
    print(f"Common pairs: {len(common)}\n")

    ciA, mA, sA = ci95_of_mean(a)
    ciB, mB, sB = ci95_of_mean(b)
    res_fold = paired_ttest(a, b)

    print("== Paired t-test (fold-wise) ==")
    print(f"{labelA}: mean={mA:.6f} ± {sA:.6f}  CI95%=[{ciA[0]:.6f}, {ciA[1]:.6f}]")
    print(f"{labelB}: mean={mB:.6f} ± {sB:.6f}  CI95%=[{ciB[0]:.6f}, {ciB[1]:.6f}]")
    print(f"Diff(A-B): mean={res_fold['mean_diff']:.6f}  CI95%=[{res_fold['ci95_diff'][0]:.6f}, {res_fold['ci95_diff'][1]:.6f}]")
    print(f"t({res_fold['n']-1})={res_fold['t']:.6f}, p(two-tailed)={res_fold['p_two']:.6e}, d={res_fold['cohen_d']:.6f}\n")

    def aggregate_by_seed(vals: np.ndarray, pairs: list[tuple[int,int]]):
        by_seed = defaultdict(list)
        for (s,f), v in zip(pairs, vals):
            by_seed[s].append(float(v))
        seeds = sorted(by_seed)
        means = np.array([np.mean(by_seed[s]) for s in seeds], dtype=float)
        return means, seeds

    a_seed, seedsA = aggregate_by_seed(a, common)
    b_seed, seedsB = aggregate_by_seed(b, common)
    assert seedsA == seedsB
    ciAs, mAs, sAs = ci95_of_mean(a_seed)
    ciBs, mBs, sBs = ci95_of_mean(b_seed)
    res_seed = paired_ttest(a_seed, b_seed)

    print("== Paired t-test (seed-wise, recommended) ==")
    print(f"{labelA} (per-seed means): mean={mAs:.6f} ± {sAs:.6f}  CI95%=[{ciAs[0]:.6f}, {ciAs[1]:.6f}]")
    print(f"{labelB} (per-seed means): mean={mBs:.6f} ± {sBs:.6f}  CI95%=[{ciBs[0]:.6f}, {ciBs[1]:.6f}]")
    print(f"Diff(A-B): mean={res_seed['mean_diff']:.6f}  CI95%=[{res_seed['ci95_diff'][0]:.6f}, {res_seed['ci95_diff'][1]:.6f}]")
    print(f"t({res_seed['n']-1})={res_seed['t']:.6f}, p(two-tailed)={res_seed['p_two']:.6e}, d={res_seed['cohen_d']:.6f}\n")

    seeds = sorted({s for s, _ in common})
    folds = sorted({f for _, f in common})
    Amap_metric = {k: (A[k].get("test_metrics_json") or {}).get(mk, np.nan) for k in common}
    Bmap_metric = {k: (B[k].get("test_metrics_json") or {}).get(mk, np.nan) for k in common}

    print_matrix_table(f"== Table — {labelA} — test.{mk} ==", seeds, folds,
                       build_matrix(Amap_metric, seeds, folds))
    print_matrix_table(f"== Table — {labelB} — test.{mk} ==", seeds, folds,
                       build_matrix(Bmap_metric, seeds, folds))

    print("Note: 'mean_fold' = mean across seeds for a given fold; 'mean_seed' = mean across folds for a given seed;")
    print("      last cell of 'mean_seed' row is the AGGREGATED MEAN across all seedxfold cells.\n")

    def flatten_all(results: dict):
        return {k: flatten_record(v) for k, v in results.items()}

    Aflat = flatten_all(A)
    Bflat = flatten_all(B)

    def keys_by_prefix(flat_map: dict, prefix: str):
        keys = set()
        for rec in flat_map.values():
            for k, v in rec.items():
                if k.startswith(prefix) and isinstance(v, (int,float)):
                    keys.add(k)
        return sorted(keys)

    def keys_eff(flat_map: dict):
        eff = []
        want = ["train_time_sec","test_time_sec","test_inf_ms_per_sample","throughput_samples_per_sec",
                "max_gpu_mem_mb","best_checkpoint_size_mb","best_epoch","n_train","n_val","n_test","num_params"]
        for k in want:
            present = any(isinstance(rec.get(k, None), (int,float)) for rec in flat_map.values())
            if present:
                eff.append(k)
        return eff

    test_keys = sorted(set(keys_by_prefix(Aflat, "test.")).union(keys_by_prefix(Bflat, "test.")))
    val_keys  = sorted(set(keys_by_prefix(Aflat, "val.")).union(keys_by_prefix(Bflat, "val.")))
    eff_keys  = sorted(set(keys_eff(Aflat)).union(keys_eff(Bflat)))

    def matrix_for_field(flat_map: dict, field: str):
        m = {}
        for (s,f) in common:
            v = flat_map.get((s,f), {}).get(field, np.nan)
            m[(s,f)] = v
        return build_matrix(m, seeds, folds)

    print("== ALL TEST METRICS (per model) ==")
    for key in test_keys:
        print_matrix_table(f"-- {labelA} — {key} --", seeds, folds, matrix_for_field(Aflat, key))
        print_matrix_table(f"-- {labelB} — {key} --", seeds, folds, matrix_for_field(Bflat, key))

    print("== ALL VALIDATION METRICS (per model) ==")
    for key in val_keys:
        print_matrix_table(f"-- {labelA} — {key} --", seeds, folds, matrix_for_field(Aflat, key))
        print_matrix_table(f"-- {labelB} — {key} --", seeds, folds, matrix_for_field(Bflat, key))

    print("== EFFICIENCY / RESOURCES / COUNTS (per model) ==")
    for key in eff_keys:
        print_matrix_table(f"-- {labelA} — {key} --", seeds, folds, matrix_for_field(Aflat, key))
        print_matrix_table(f"-- {labelB} — {key} --", seeds, folds, matrix_for_field(Bflat, key))

    def numeric_fields_in_common():
        keysA = set()
        keysB = set()
        for rec in Aflat.values():
            for k, v in rec.items():
                if isinstance(v, (int,float)):
                    keysA.add(k)
        for rec in Bflat.values():
            for k, v in rec.items():
                if isinstance(v, (int,float)):
                    keysB.add(k)
        return sorted(keysA.intersection(keysB))

    print("== A-B DIFFERENCE TABLES (foldxseed) ==")
    for key in numeric_fields_in_common():
        diff_map = {}
        for (s,f) in common:
            va = Aflat.get((s,f), {}).get(key, np.nan)
            vb = Bflat.get((s,f), {}).get(key, np.nan)
            if isinstance(va, (int,float)) and isinstance(vb, (int,float)):
                diff_map[(s,f)] = float(va) - float(vb)
            else:
                diff_map[(s,f)] = np.nan
        print_matrix_table(f"-- Diff(A-B) — {key} --", seeds, folds, build_matrix(diff_map, seeds, folds))

    if args.print_hparams:
        def maybe_print_hparams(root: Path, label: str):
            hp_root = root / "hyperparams_used.json"
            if hp_root.exists():
                print(f"== hyperparams_used.json — {label} ==")
                try:
                    print(hp_root.read_text(encoding='utf-8').strip())
                except Exception as e:
                    print(f"[warn] could not read hyperparams: {e}")
                print()
        maybe_print_hparams(rootA, labelA)
        maybe_print_hparams(rootB, labelB)

    if out_fh:
        sys.stdout = stdout_backup
        out_fh.close()

if __name__ == "__main__":
    main()