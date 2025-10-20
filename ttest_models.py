#!/usr/bin/env python3
import argparse, os, json, math, sys, csv, re
from pathlib import Path
from collections import defaultdict
import numpy as np

try:
    from scipy import stats
except Exception as e:
    raise SystemExit("scipy is required. Try: pip install scipy") from e

# ---- Candidatas de métrica para autodetecção ----
CAND_KEYS = [
    "test_accuracy","accuracy","acc",
    "test_f1","f1","f1_score",
    "test_precision","precision","prec",
    "test_recall","recall","rec",
    "test_loss","loss"
]

# ---- Tee p/ gravar stdout em arquivo (--out) ----
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
    def flush(self):
        for f in self.files:
            f.flush()

# ---- Utilitários de parsing ----
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
    # Tenta json normal
    try:
        return json.loads(jstr)
    except Exception:
        pass
    # Normaliza: troca aspas simples por duplas e NaN/nan por null
    j2 = jstr.replace("'", '"')
    j2 = re.sub(r"\bNaN\b", "null", j2, flags=re.IGNORECASE)
    try:
        return json.loads(j2)
    except Exception:
        return {}

def parse_result_file(path: Path):
    """
    Lê um resultados_fold.txt e retorna:
      seed, fold, data_dict (inclui val_metrics_json e test_metrics_json como dicts)
    """
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
                # tenta numérico/bool; se for lista textual, guarda como string
                data[key] = _to_number_or_str(val)
    return seed, fold, data

def pick_metric_key(sample_metrics: dict, prefer: str|None):
    if prefer and prefer in sample_metrics:
        return prefer
    for k in CAND_KEYS:
        if k in sample_metrics and isinstance(sample_metrics[k], (int,float)):
            return k
    for k,v in sample_metrics.items():
        if isinstance(v, (int,float)):
            return k
    raise ValueError("Não foi possível identificar uma métrica numérica em test_metrics_json. "
                     f"Chaves vistas: {list(sample_metrics.keys())[:10]}")

def gather_results(root: Path):
    out = {}
    for p in root.rglob("resultados_fold.txt"):
        s,f,dat = parse_result_file(p)
        if s is None or f is None:
            continue
        out[(s,f)] = dat
    return out

# ---- Rótulo a partir de tmodel (hyperparams_used.json) ----
def infer_model_label(root: Path, manual_label: str|None = None) -> str:
    if manual_label:
        return manual_label
    candidates = []
    hp_root = root / "hyperparams_used.json"
    if hp_root.exists():
        candidates.append(hp_root)
    if not candidates:
        for p in root.glob("**/hyperparams_used.json"):
            candidates.append(p); break
    for hp in candidates:
        try:
            data = json.loads(hp.read_text(encoding="utf-8"))
            for key in ("tmodel", "TMODEL", "model", "name_model"):
                if key in data and isinstance(data[key], str) and data[key].strip():
                    return data[key].strip()
        except Exception:
            pass
    return root.name

# ---- Estatística ----
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

def aggregate_by_seed(vals: np.ndarray, pairs: list[tuple[int,int]]):
    by_seed = defaultdict(list)
    for (s,f), v in zip(pairs, vals):
        by_seed[s].append(float(v))
    seeds = sorted(by_seed)
    means = np.array([np.mean(by_seed[s]) for s in seeds], dtype=float)
    return means, seeds

# ---- Tabelas por modelo (folds × seeds) ----
def build_matrix(vals_map: dict, seeds: list[int], folds: list[int]):
    M = np.full((len(folds), len(seeds)), np.nan)
    for i, f in enumerate(folds):
        for j, s in enumerate(seeds):
            v = vals_map.get((s, f), np.nan)
            if isinstance(v, (int,float)):
                M[i, j] = float(v)
    return M

def fmt_num(x):
    return "NA".rjust(8) if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{x:8.4f}"

def print_model_table(label: str, metric_name: str, seeds: list[int], folds: list[int], M: np.ndarray):
    header = ["fold\\seed"] + [f"s{int(s)}" for s in seeds] + ["mean_fold"]
    widths = [max(9, len(h)) for h in header]
    def print_row(cells):
        print(" | ".join(str(c).rjust(w) for c, w in zip(cells, widths)))
    print(f"=== Tabela — {label} — {metric_name} ===")
    print_row(header)
    print("-" * (sum(widths) + 3*(len(widths)-1)))
    row_means = np.nanmean(M, axis=1) if M.size else np.array([])
    for i, f in enumerate(folds):
        cells = [f"fold_{f}"] + [fmt_num(M[i, j]) for j in range(len(seeds))] + [fmt_num(row_means[i])]
        print_row(cells)
    col_means = np.nanmean(M, axis=0) if M.size else np.array([])
    agg = float(np.nanmean(M)) if M.size else float("nan")
    print("-" * (sum(widths) + 3*(len(widths)-1)))
    tail = ["mean_seed"] + [fmt_num(col_means[j]) for j in range(len(seeds))] + [fmt_num(agg)]
    print_row(tail)
    print()

# ---- Flatten para CSV ----
NUMERIC_KEEP = {"best_epoch","train_time_sec","test_time_sec","test_inf_ms_per_sample",
                "throughput_samples_per_sec","max_gpu_mem_mb","best_checkpoint_size_mb",
                "n_train","n_val","n_test","num_params"}
CATEG_KEEP = {"balance_mode","class_weights_injected"}  # strings/bools úteis

def flatten_record(data: dict):
    flat = {}
    # campos simples
    for k, v in data.items():
        if k in ("val_metrics_json","test_metrics_json","Seed","Fold"):
            continue
        if k in NUMERIC_KEEP:
            flat[k] = _to_number_or_str(v)
        elif k in CATEG_KEEP:
            flat[k] = v
    # métricas
    for pref, key in (("val","val_metrics_json"), ("test","test_metrics_json")):
        md = data.get(key, {}) or {}
        for mk, mv in md.items():
            val = _to_number_or_str(mv)
            flat[f"{pref}.{mk}"] = val
    return flat

def save_model_csv(root: Path, label: str, results: dict, out_dir: Path):
    # union de colunas
    seeds = sorted({s for s,_ in results.keys()})
    folds = sorted({f for _,f in results.keys()})
    cols = ["seed","fold"]
    all_keys = set()
    for k, dat in results.items():
        all_keys |= set(flatten_record(dat).keys())
    cols += sorted(all_keys)

    out_path = out_dir / f"{label}_per_fold.csv"
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(cols)
        for (s,f) in sorted(results.keys()):
            row = {"seed": s, "fold": f}
            row.update(flatten_record(results[(s,f)]))
            wr.writerow([row.get(c, "") for c in cols])
    return out_path

def is_number(x):
    return isinstance(x, (int,float)) and not (isinstance(x,float) and np.isnan(x))

def save_paired_csv(labelA: str, A: dict, labelB: str, B: dict, out_dir: Path):
    common = sorted(set(A).intersection(B))
    # descubra colunas numéricas em comum (após flatten)
    keysA = set()
    keysB = set()
    cacheA = {}
    cacheB = {}
    for k in common:
        fa = flatten_record(A[k]); fb = flatten_record(B[k])
        cacheA[k] = fa; cacheB[k] = fb
        for kk, vv in fa.items():
            if is_number(vv): keysA.add(kk)
        for kk, vv in fb.items():
            if is_number(vv): keysB.add(kk)
    inter = sorted(keysA.intersection(keysB))

    cols = ["seed","fold"]
    for kk in inter:
        cols += [f"{kk}__{labelA}", f"{kk}__{labelB}", f"{kk}__diff"]
    out_path = out_dir / f"paired_{labelA}_vs_{labelB}.csv"
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(cols)
        for (s,f) in common:
            row = {"seed": s, "fold": f}
            fa = cacheA[(s,f)]; fb = cacheB[(s,f)]
            for kk in inter:
                va = fa.get(kk, "")
                vb = fb.get(kk, "")
                vd = ""
                if is_number(va) and is_number(vb):
                    vd = float(va) - float(vb)
                row[f"{kk}__{labelA}"] = va
                row[f"{kk}__{labelB}"] = vb
                row[f"{kk}__diff"] = vd
            wr.writerow([row.get(c, "") for c in cols])
    return out_path

# ---- Main ----
def main():
    ap = argparse.ArgumentParser(description="t-teste pareado entre dois modelos a partir de resultados_fold.txt")
    ap.add_argument("--modelA", required=True, help="Pasta do modelo A (ex.: modelos_kf/soja_convnext_t)")
    ap.add_argument("--modelB", required=True, help="Pasta do modelo B (ex.: modelos_kf/soja_vector)")
    ap.add_argument("--metric", default=None, help="Chave da métrica (ex.: accuracy, f1, test_loss). Se omitido, autodetecta.")
    ap.add_argument("--labelA", default=None, help="Rótulo manual para o modelo A (sobrepõe 'tmodel')")
    ap.add_argument("--labelB", default=None, help="Rótulo manual para o modelo B (sobrepõe 'tmodel')")
    ap.add_argument("--out", default=None, help="Se informado, salva a saída completa em um arquivo .txt")
    ap.add_argument("--dump_dir", default=None, help="Diretório onde serão salvos os CSVs com TODOS os parâmetros")
    args = ap.parse_args()

    stdout_backup = sys.stdout

    # saída opcional em arquivo
    out_fh = None
    if args.out:
        out_dir = os.path.dirname(os.path.expanduser(args.out)) or "."
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        out_fh = open(os.path.expanduser(args.out), "w", encoding="utf-8")
        sys.stdout = Tee(sys.stdout, out_fh)

    rootA = Path(os.path.expanduser(args.modelA)).resolve()
    rootB = Path(os.path.expanduser(args.modelB)).resolve()
    if not rootA.exists() or not rootB.exists():
        raise SystemExit(f"Caminhos inválidos:\n A={rootA}\n B={rootB}")

    # rótulos a partir de tmodel (ou manual)
    labelA = infer_model_label(rootA, args.labelA)
    labelB = infer_model_label(rootB, args.labelB)

    A = gather_results(rootA)
    B = gather_results(rootB)
    common = sorted(set(A).intersection(B))
    if not common:
        raise SystemExit("Não há pares (seed, fold) em comum entre as duas pastas. "
                         "Verifique se os dois modelos foram treinados com as mesmas seeds e folds.")

    # Detecta métrica
    # usa A como referência; se não existir em B, tenta conciliar
    mk = args.metric or pick_metric_key(A[common[0]].get("test_metrics_json", {}), args.metric)
    if mk not in B[common[0]].get("test_metrics_json", {}):
        keysA = set((A[common[0]].get("test_metrics_json", {}) or {}).keys())
        keysB = set((B[common[0]].get("test_metrics_json", {}) or {}).keys())
        inter = [k for k in CAND_KEYS if k in keysA and k in keysB]
        if inter:
            mk = inter[0]
        else:
            raise SystemExit(f"A métrica '{mk}' não está presente em ambos.\n"
                             f"Exemplos A={list(keysA)[:6]} | B={list(keysB)[:6]}")

    # Vetores pareados para t-teste (apenas a métrica escolhida)
    a_vals, b_vals = [], []
    for k in common:
        va = (A[k].get("test_metrics_json") or {}).get(mk, None)
        vb = (B[k].get("test_metrics_json") or {}).get(mk, None)
        if isinstance(va, (int,float)) and isinstance(vb, (int,float)):
            a_vals.append(float(va))
            b_vals.append(float(vb))
    a = np.array(a_vals, dtype=float)
    b = np.array(b_vals, dtype=float)

    print(f"Usando métrica: {mk}")
    print(f"Pares (seed, fold) em comum: {len(common)}")
    print(f"Comparando {labelA} (A) vs {labelB} (B) — Diferença reportada é A−B\n")

    # ---- Estatísticas pareadas ----
    ciA, mA, sA = ci95_of_mean(a)
    ciB, mB, sB = ci95_of_mean(b)
    res_fold = paired_ttest(a, b)

    print("=== Comparação fold-a-fold (pareado) ===")
    print(f"{labelA}: média={mA:.4f} ± {sA:.4f}  IC95%=[{ciA[0]:.4f}, {ciA[1]:.4f}]")
    print(f"{labelB}: média={mB:.4f} ± {sB:.4f}  IC95%=[{ciB[0]:.4f}, {ciB[1]:.4f}]")
    print(f"Diff(A-B): média={res_fold['mean_diff']:.4f}  IC95%=[{res_fold['ci95_diff'][0]:.4f}, {res_fold['ci95_diff'][1]:.4f}]")
    print(f"t({res_fold['n']-1})={res_fold['t']:.3f}, p(bicaudal)={res_fold['p_two']:.3e}, d={res_fold['cohen_d']:.3f}\n")

    a_seed, seedsA = aggregate_by_seed(a, common)
    b_seed, seedsB = aggregate_by_seed(b, common)
    assert seedsA == seedsB
    ciAs, mAs, sAs = ci95_of_mean(a_seed)
    ciBs, mBs, sBs = ci95_of_mean(b_seed)
    res_seed = paired_ttest(a_seed, b_seed)

    print("=== Comparação seed-a-seed (pareado; recomendado) ===")
    print(f"{labelA} (médias por seed): média={mAs:.4f} ± {sAs:.4f}  IC95%=[{ciAs[0]:.4f}, {ciAs[1]:.4f}]")
    print(f"{labelB} (médias por seed): média={mBs:.4f} ± {sBs:.4f}  IC95%=[{ciBs[0]:.4f}, {ciBs[1]:.4f}]")
    print(f"Diff(A-B): média={res_seed['mean_diff']:.4f}  IC95%=[{res_seed['ci95_diff'][0]:.4f}, {res_seed['ci95_diff'][1]:.4f}]")
    print(f"t({res_seed['n']-1})={res_seed['t']:.3f}, p(bicaudal)={res_seed['p_two']:.3e}, d={res_seed['cohen_d']:.3f}\n")

    # ---- Tabelas por modelo (folds × seeds) ----
    seeds = sorted({s for s, _ in common})
    folds = sorted({f for _, f in common})
    Amap_metric = {k: (A[k].get("test_metrics_json") or {}).get(mk, np.nan) for k in common}
    Bmap_metric = {k: (B[k].get("test_metrics_json") or {}).get(mk, np.nan) for k in common}
    MA = build_matrix(Amap_metric, seeds, folds)
    MB = build_matrix(Bmap_metric, seeds, folds)

    print_model_table(labelA, mk, seeds, folds, MA)
    print_model_table(labelB, mk, seeds, folds, MB)

    print("Nota: 'mean_fold' = média entre seeds no mesmo fold; 'mean_seed' = média entre folds na mesma seed;")
    print("      célula final de 'mean_seed' é a MÉDIA AGREGADA (todos os folds × seeds).\n")

    # ---- Dump completo dos parâmetros em CSVs ----
    if args.dump_dir:
        out_dir = Path(os.path.expanduser(args.dump_dir)).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        pathA = save_model_csv(rootA, labelA, A, out_dir)
        pathB = save_model_csv(rootB, labelB, B, out_dir)
        pathP = save_paired_csv(labelA, A, labelB, B, out_dir)
        print(f"[CSV] Salvo parâmetros de {labelA}: {pathA}")
        print(f"[CSV] Salvo parâmetros de {labelB}: {pathB}")
        print(f"[CSV] Salvo comparação pareada:  {pathP}\n")

    if out_fh:
        sys.stdout = stdout_backup
        out_fh.close()

if __name__ == "__main__":
    main()
