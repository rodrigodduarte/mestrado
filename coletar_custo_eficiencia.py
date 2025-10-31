#!/usr/bin/env python3
# coletar_custo_eficiencia.py
# Agrega SOMENTE métricas de custo/eficiência de todos os benchmark_result.txt
# e escreve tudo em UM arquivo: tabelas_custo_eficiencia.md

import os
import argparse
import math
from collections import defaultdict, OrderedDict

# --------- Configuração das chaves que vamos coletar (somente custo/eficiência) ----------
# Metadados mínimos para indexar as tabelas:
META_KEYS = OrderedDict([
    ("dataset", str),
    ("tmodel", str),
    ("fold_idx", int),
])

# Métricas de custo/eficiência (somente estas serão agregadas):
COST_KEYS = OrderedDict([
    ("train_time_sec", float),
    ("test_time_sec", float),
    ("test_inf_ms_per_sample", float),
    ("throughput_samples_per_sec", float),
    ("max_gpu_mem_mb_train", float),
    ("max_gpu_mem_mb_test", float),
    ("model_size_mb", float),
    ("num_params_trainable", float),  # pode vir como int, tratamos como float na impressão
    # campos que podem existir em alguns formatos:
    ("best_checkpoint_size_mb", float),
    ("epoch_trained", float),  # útil como diagnóstico; não é “custo” mas ajuda
])

ALL_KEYS = list(META_KEYS.keys()) + list(COST_KEYS.keys())

# --------- Utilidades ---------
def _to_number(x, typ):
    s = str(x).strip()
    if s.lower() in ("none", "nan"):
        return math.nan
    try:
        return typ(s)
    except Exception:
        try:
            # às vezes floats vêm como "123.0\n" ou com espaços
            return typ(s.replace(",", "."))
        except Exception:
            return math.nan

def parse_benchmark_result_txt(path):
    """
    Lê um benchmark_result.txt e extrai apenas META_KEYS + COST_KEYS.
    Ignora linhas de métricas de acurácia/prec/rec/f1/loss.
    """
    record = {k: (math.nan if t in (int, float) else None) for k, t in META_KEYS.items()}
    record.update({k: math.nan for k in COST_KEYS.keys()})

    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if ":" not in line:
                    continue
                k, v = line.split(":", 1)
                k = k.strip()
                v = v.strip()
                if k in META_KEYS:
                    typ = META_KEYS[k]
                    if typ is str:
                        record[k] = v
                    else:
                        record[k] = _to_number(v, typ)
                elif k in COST_KEYS:
                    typ = COST_KEYS[k]
                    record[k] = _to_number(v, typ)
    except Exception as e:
        print(f"[WARN] Falha lendo {path}: {e}")

    # dataset/tmodel podem às vezes ser inferidos do caminho
    if not record.get("dataset"):
        # tenta inferir do diretório pai do diretório raiz do fold
        # .../estatisticas/<dataset>_<tmodel>/fold_k/benchmark_result.txt
        parts = os.path.normpath(path).split(os.sep)
        try:
            idx = parts.index("estatisticas")
            run_dir = parts[idx+1]  # ex: D2_convnext_t
            if "_" in run_dir:
                record["dataset"] = run_dir.split("_", 1)[0]
                record["tmodel"]  = run_dir.split("_", 1)[1]
        except Exception:
            pass

    return record

def walk_benchmark_results(root):
    files = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn == "benchmark_result.txt":
                files.append(os.path.join(dirpath, fn))
    return files

def mean_std(values):
    vals = [float(x) for x in values if isinstance(x, (int, float)) and not math.isnan(x)]
    n = len(vals)
    if n == 0:
        return (math.nan, math.nan, 0)
    m = sum(vals) / n
    if n == 1:
        return (m, 0.0, 1)
    var = sum((x - m) ** 2 for x in vals) / (n - 1)
    return (m, math.sqrt(var), n)

def fmt_mean_std(m, s):
    if math.isnan(m):
        return "N/A"
    return f"{m:.6f} ± {s:.6f}"

def build_tables(records):
    """
    Constrói:
      - summary[(dataset, tmodel)][metric] = (mean, std, n)
      - long_rows: linhas completas por fold
    """
    # agrupa por dataset/tmodel
    groups = defaultdict(list)
    for r in records:
        ds = r.get("dataset") or "UNKNOWN"
        tm = r.get("tmodel") or "UNKNOWN"
        groups[(ds, tm)].append(r)

    summary = {}
    for (ds, tm), rows in groups.items():
        metrics = {}
        for k in COST_KEYS.keys():
            m, s, n = mean_std([row.get(k) for row in rows])
            metrics[k] = (m, s, n)
        summary[(ds, tm)] = metrics

    # tabela longa (linhas por fold)
    long_rows = []
    for r in records:
        row = OrderedDict()
        for k in META_KEYS.keys():
            row[k] = r.get(k)
        for k in COST_KEYS.keys():
            row[k] = r.get(k)
        long_rows.append(row)

    return summary, long_rows

def write_markdown(output_path, summary, long_rows):
    # ordenação consistente
    keys_sorted = sorted(summary.keys(), key=lambda kt: (str(kt[0]), str(kt[1])))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Tabelas de Custo/Eficiência (agregadas a partir de benchmark_result.txt)\n\n")
        f.write("> Este arquivo contém **somente** métricas de custo/eficiência. "
                "Métricas de desempenho (acurácia/F1/etc.) foram omitidas conforme solicitado.\n\n")

        # ---- Sumário por (dataset, tmodel) ----
        f.write("## Sumário por dataset e modelo (média ± desvio)\n\n")
        for (ds, tm) in keys_sorted:
            f.write(f"### {ds} — {tm}\n\n")
            f.write("| Métrica | Média ± DP | n |\n")
            f.write("|---|---:|---:|\n")
            metrics = summary[(ds, tm)]
            for k in COST_KEYS.keys():
                m, s, n = metrics.get(k, (math.nan, math.nan, 0))
                f.write(f"| `{k}` | {fmt_mean_std(m, s)} | {n} |\n")
            f.write("\n")

        # ---- Tabela longa (todas as linhas por fold) ----
        f.write("## Tabela longa (todas as observações por fold)\n\n")
        # cabeçalho
        headers = list(META_KEYS.keys()) + list(COST_KEYS.keys())
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in long_rows:
            cells = []
            for h in headers:
                v = row.get(h)
                if isinstance(v, float):
                    cells.append("" if math.isnan(v) else f"{v:.6f}")
                else:
                    cells.append("" if v is None else str(v))
            f.write("| " + " | ".join(cells) + " |\n")

def main():
    ap = argparse.ArgumentParser(description="Agrega métricas de custo/eficiência dos benchmark_result.txt em um único arquivo .md")
    ap.add_argument("--root", type=str, default=".", help="Diretório raiz para procurar benchmark_result.txt (padrão: diretório atual)")
    ap.add_argument("--out", type=str, default="tabelas_custo_eficiencia.md", help="Nome do arquivo único de saída (.md)")
    args = ap.parse_args()

    files = walk_benchmark_results(args.root)
    if not files:
        print(f"[ERRO] Nenhum 'benchmark_result.txt' encontrado sob: {args.root}")
        return

    records = []
    for p in sorted(files):
        rec = parse_benchmark_result_txt(p)
        # somente aceita se tiver dataset e tmodel
        if not rec.get("dataset") or not rec.get("tmodel"):
            print(f"[WARN] Ignorando (faltam metadados dataset/tmodel): {p}")
            continue
        records.append(rec)

    if not records:
        print("[ERRO] Não há registros válidos após o parsing.")
        return

    summary, long_rows = build_tables(records)
    write_markdown(args.out, summary, long_rows)

    print(f"[OK] Processados {len(records)} arquivos.")
    print(f"[OK] Saída única escrita em: {os.path.abspath(args.out)}")

if __name__ == "__main__":
    main()
