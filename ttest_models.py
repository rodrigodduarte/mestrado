#!/usr/bin/env python3
import argparse, os, json, math, sys
from pathlib import Path
from collections import defaultdict
import numpy as np

try:
    from scipy import stats
except Exception as e:
    raise SystemExit("scipy is required. Try: pip install scipy") from e

# ---- Preferências de chaves de métrica (autodetecção) ----
CAND_KEYS = [
    "test_accuracy","accuracy","acc",
    "test_f1","f1","f1_score",
    "test_precision","precision","prec",
    "test_recall","recall","rec",
    "test_loss","loss"
]

# ---- Utilitário para salvar stdout opcionalmente em arquivo (--out) ----
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
    def flush(self):
        for f in self.files:
            f.flush()

# ---- Parsers dos resultados por fold ----
def parse_result_file(path: Path):
    seed = fold = None
    test_metrics = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("Seed:"):
                try:
                    seed = int(line.split(":",1)[1].strip())
                except:
                    pass
            elif line.startswith("Fold:"):
                try:
                    fold = int(line.split(":",1)[1].strip())
                except:
                    pass
            elif line.startswith("test_metrics_json:"):
                j = line.split(":",1)[1].strip()
                try:
                    test_metrics = json.loads(j)
                except json.JSONDecodeError:
                    # reparo simples (caso tenha aspas simples)
                    j2 = j.replace("'", '"')
                    try:
                        test_metrics = json.loads(j2)
                    except Exception:
                        test_metrics = {}
    return seed, fold, test_metrics

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
        s,f,tm = parse_result_file(p)
        if s is None or f is None or not tm:
            continue
        out[(s,f)] = tm
    return out

# ---- Inferência de rótulo a partir do hiperparâmetro tmodel ----
def infer_model_label(root: Path, manual_label: str|None = None) -> str:
    """
    Retorna o rótulo a ser impresso.
    Prioridade:
      1) manual_label (se fornecido por --labelA/--labelB)
      2) valor de 'tmodel' (ou 'TMODEL') em hyperparams_used.json
      3) nome da pasta root
    """
    if manual_label:
        return manual_label
    # tenta achar hyperparams_used.json na raiz ou logo abaixo
    candidates = []
    hp_root = root / "hyperparams_used.json"
    if hp_root.exists():
        candidates.append(hp_root)
    # fallback: procura o primeiro arquivo com esse nome abaixo da raiz
    if not candidates:
        for p in root.glob("**/hyperparams_used.json"):
            candidates.append(p)
            break
    for hp in candidates:
        try:
            data = json.loads(hp.read_text(encoding="utf-8"))
            for key in ("tmodel", "TMODEL", "model", "name_model"):
                if key in data and isinstance(data[key], str) and data[key].strip():
                    return data[key].strip()
        except Exception:
            pass
    return root.name  # fallback final

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

# ---- Main ----
def main():
    ap = argparse.ArgumentParser(description="t-teste pareado entre dois modelos a partir de resultados_fold.txt")
    ap.add_argument("--modelA", required=True, help="Pasta do modelo A (ex.: modelos_kf/soja_convnext_t)")
    ap.add_argument("--modelB", required=True, help="Pasta do modelo B (ex.: modelos_kf/soja_vector)")
    ap.add_argument("--metric", default=None, help="Chave da métrica (ex.: accuracy, f1, test_loss). Se omitido, autodetecta.")
    ap.add_argument("--labelA", default=None, help="Rótulo manual para o modelo A (sobrepõe 'tmodel')")
    ap.add_argument("--labelB", default=None, help="Rótulo manual para o modelo B (sobrepõe 'tmodel')")
    ap.add_argument("--out", default=None, help="Se informado, salva a saída completa em um arquivo .txt")
    args = ap.parse_args()

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
    mk = args.metric or pick_metric_key(A[common[0]], args.metric)
    if mk not in B[common[0]]:
        # tenta conciliar por chaves candidatas
        keysA = set(A[common[0]].keys())
        keysB = set(B[common[0]].keys())
        inter = [k for k in CAND_KEYS if k in keysA and k in keysB]
        if inter:
            mk = inter[0]
        else:
            raise SystemExit(f"A métrica '{mk}' não está presente em ambos. "
                             f"Exemplos A={list(keysA)[:6]} | B={list(keysB)[:6]}")

    a_vals, b_vals = [], []
    for k in common:
        va = A[k].get(mk, None)
        vb = B[k].get(mk, None)
        if isinstance(va, (int,float)) and isinstance(vb, (int,float)):
            a_vals.append(float(va))
            b_vals.append(float(vb))
    a = np.array(a_vals, dtype=float)
    b = np.array(b_vals, dtype=float)

    print(f"Usando métrica: {mk}")
    print(f"Pares (seed, fold) em comum: {len(common)}")
    print(f"Comparando {labelA} (A) vs {labelB} (B) — Diferença reportada é A−B\n")

    # Fold-a-fold
    ciA, mA, sA = ci95_of_mean(a)
    ciB, mB, sB = ci95_of_mean(b)
    res_fold = paired_ttest(a, b)

    print("=== Comparação fold-a-fold (pareado) ===")
    print(f"{labelA}: média={mA:.4f} ± {sA:.4f}  IC95%=[{ciA[0]:.4f}, {ciA[1]:.4f}]")
    print(f"{labelB}: média={mB:.4f} ± {sB:.4f}  IC95%=[{ciB[0]:.4f}, {ciB[1]:.4f}]")
    print(f"Diff(A-B): média={res_fold['mean_diff']:.4f}  IC95%=[{res_fold['ci95_diff'][0]:.4f}, {res_fold['ci95_diff'][1]:.4f}]")
    print(f"t({res_fold['n']-1})={res_fold['t']:.3f}, p(bicaudal)={res_fold['p_two']:.3e}, d={res_fold['cohen_d']:.3f}\n")

    # Seed-a-seed (recomendado)
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

    print("=== Comparação seed-a-seed (pareado; recomendado) ===")
    print(f"{labelA} (médias por seed): média={mAs:.4f} ± {sAs:.4f}  IC95%=[{ciAs[0]:.4f}, {ciAs[1]:.4f}]")
    print(f"{labelB} (médias por seed): média={mBs:.4f} ± {sBs:.4f}  IC95%=[{ciBs[0]:.4f}, {ciBs[1]:.4f}]")
    print(f"Diff(A-B): média={res_seed['mean_diff']:.4f}  IC95%=[{res_seed['ci95_diff'][0]:.4f}, {res_seed['ci95_diff'][1]:.4f}]")
    print(f"t({res_seed['n']-1})={res_seed['t']:.3f}, p(bicaudal)={res_seed['p_two']:.3e}, d={res_seed['cohen_d']:.3f}\n")

    print("Nota: para teste unilateral (ex.: A>B), p_unicaudal = p_bicaudal/2 quando a diferença tem o sinal esperado.")

    if out_fh:
        out_fh.close()

if __name__ == "__main__":
    main()
