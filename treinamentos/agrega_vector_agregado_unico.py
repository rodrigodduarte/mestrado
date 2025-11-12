#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import re, json, sys, statistics as stats

METRICAS_TESTE = ["test_accuracy", "test_f1", "test_precision", "test_recall", "test_loss"]
HEADER_BLOCK = re.compile(r"^=+\s+fold_(\d+)/resultados_fold\.txt\s+=+\s*$", re.MULTILINE)

def find_treinamentos_from_cwd() -> Path | None:
    cwd = Path.cwd()
    for p in [cwd] + list(cwd.parents):
        t = p / "treinamentos"
        if (t / "estatisticas").is_dir():
            return t
    return None

def parse_estatisticas_completas(path_txt: Path):
    txt = path_txt.read_text(encoding="utf-8", errors="replace")
    parts = HEADER_BLOCK.split(txt)
    folds = []
    for i in range(1, len(parts), 2):
        fold_id = parts[i]
        bloco = parts[i+1] if i+1 < len(parts) else ""
        reg = {"fold_idx": int(fold_id)}
        for line in bloco.splitlines():
            line = line.strip()
            if not line or line.startswith("===="):
                continue
            if line.startswith("test_metrics_json:"):
                blob = line.split("test_metrics_json:", 1)[1].strip()
                if not blob.endswith("}"):
                    m = re.search(r"test_metrics_json:\s*(\{.*?\})", bloco.replace("\n"," "))
                    if m: blob = m.group(1)
                try:
                    reg["test_metrics_json"] = json.loads(blob)
                except Exception:
                    reg["test_metrics_json"] = {}
        reg.setdefault("test_metrics_json", {})
        folds.append(reg)
    return folds

def fmt_mean_std(vals):
    if not vals: return "n/a"
    if len(vals) == 1: return f"{vals[0]:.6f} ± 0.000000"
    return f"{stats.mean(vals):.6f} ± {stats.pstdev(vals):.6f}"

def main():
    # Detecta 'treinamentos' automaticamente (ou aceite um caminho opcional)
    args = sys.argv[1:]
    if len(args) == 0:
        T = find_treinamentos_from_cwd()
        if T is None:
            print("ERRO: não encontrei 'treinamentos/estatisticas'. "
                  "Use: python agrega_vector_agregado_unico.py /caminho/para/treinamentos",
                  file=sys.stderr)
            sys.exit(2)
    else:
        T = Path(args[0]).expanduser().resolve()

    EST = T / "estatisticas"
    if not EST.is_dir():
        print(f"ERRO: diretório não encontrado: {EST}", file=sys.stderr)
        sys.exit(2)

    # Localiza diretórios *_vector (um por dataset)
    vec_dirs = sorted(p for p in EST.iterdir() if p.is_dir() and p.name.endswith("_vector"))

    blocos = []
    processados = 0

    for d in vec_dirs:
        dataset = d.name[:-7]  # remove "_vector"
        src = d / "estatisticas_completas.txt"
        if not src.is_file():
            continue

        folds = parse_estatisticas_completas(src)
        if not folds:
            continue

        col = {k: [] for k in METRICAS_TESTE}
        for r in folds:
            tm = r.get("test_metrics_json", {})
            for k in METRICAS_TESTE:
                v = tm.get(k)
                if isinstance(v, (int, float)):
                    col[k].append(float(v))

        linhas = []
        linhas.append(f"=== {dataset} / vector ===")
        linhas.append(f"(fonte: {src})")
        linhas.append("== Desempenho de teste (mean ± std) ==")
        linhas.append(f"Acurácia  : {fmt_mean_std(col['test_accuracy'])}")
        linhas.append(f"F1 macro  : {fmt_mean_std(col['test_f1'])}")
        linhas.append(f"Precisão  : {fmt_mean_std(col['test_precision'])}")
        linhas.append(f"Recall    : {fmt_mean_std(col['test_recall'])}")
        linhas.append(f"Loss      : {fmt_mean_std(col['test_loss'])}")
        blocos.append("\n".join(linhas))
        processados += 1

    if processados == 0:
        print("Nenhum estatisticas/*_vector/estatisticas_completas.txt válido encontrado.", file=sys.stderr)
        sys.exit(1)

    OUT = EST / "vector_agregado_desempenho.txt"  # <- nome fixo solicitado
    sep = "\n\n" + ("-" * 72) + "\n\n"
    OUT.write_text(sep.join(blocos) + "\n", encoding="utf-8")

    print(f"[OK] Gerado: {OUT}")
    print(f"[OK] Datasets processados: {processados}")

if __name__ == "__main__":
    main()
