#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import re
import sys

ROOT = Path.cwd()                 # rode em .../treinamentos
ESTAT_DIR = ROOT / "estatisticas" # onde estão as pastas dos modelos

HEADER_RE = re.compile(r"^===\s*RESUMO\s+(?P<dataset>.+?)\s*/\s*(?P<modelo>.+?)\s*===\s*$")
SECAO_DESEMPENHO = "== Desempenho de teste =="
SECAO_CUSTO = "== Custo computacional =="

def parse_resumo(texto: str):
    lines = texto.splitlines()
    dataset = modelo = ""
    if lines:
        m = HEADER_RE.match(lines[0].strip())
        if m:
            dataset = m.group("dataset").strip()
            modelo = m.group("modelo").strip()

    def extract_block(lines, start_marker, next_markers):
        start_idx = None
        for i, ln in enumerate(lines):
            if ln.strip() == start_marker:
                start_idx = i
                break
        if start_idx is None:
            return ""
        end_idx = len(lines)
        for i in range(start_idx + 1, len(lines)):
            if lines[i].strip() in next_markers:
                end_idx = i
                break
        return "\n".join(lines[start_idx:end_idx]).strip()

    bloco_desempenho = extract_block(lines, SECAO_DESEMPENHO, {SECAO_CUSTO})
    bloco_custo = extract_block(lines, SECAO_CUSTO, set())

    return {
        "dataset": dataset,
        "modelo": modelo,
        "bloco_desempenho": bloco_desempenho,
        "bloco_custo": bloco_custo,
        "conteudo_completo": texto.strip(),
    }

def main():
    if len(sys.argv) < 2:
        print("Uso: python agrega_resumos.py <DATASET>")
        print("Ex.: python agrega_resumos.py D2 | swedish | flavia")
        sys.exit(2)

    dataset_arg = sys.argv[1].strip()           # preserva caixa para nome do diretório
    dataset_alvo = dataset_arg.lower()          # para filtrar
    if not ESTAT_DIR.is_dir():
        print(f"ERRO: diretório não encontrado: {ESTAT_DIR}")
        sys.exit(2)

    # Subpastas do tipo D2_convnext_t, swedish_triple, etc.
    subdirs = sorted([p for p in ESTAT_DIR.iterdir() if p.is_dir() and "_" in p.name])

    agreg_tudo = []
    agreg_desempenho = []
    agreg_custo = []

    encontrados = 0
    for d in subdirs:
        nome = d.name
        ds_prefix = nome.split("_", 1)[0].lower()
        if ds_prefix != dataset_alvo:
            continue

        resumo_path = d / "resumo.txt"
        if not resumo_path.is_file():
            continue

        texto = resumo_path.read_text(encoding="utf-8", errors="replace")
        info = parse_resumo(texto)

        dataset = info["dataset"] or ds_prefix
        modelo = info["modelo"] or nome.split("_", 1)[1]

        bloco_tudo = []
        bloco_tudo.append(f"=== {dataset} / {modelo} ===")
        bloco_tudo.append(f"(fonte: {resumo_path})\n")
        bloco_tudo.append(info["conteudo_completo"])
        agreg_tudo.append("\n".join(bloco_tudo).strip())

        if info["bloco_desempenho"]:
            agreg_desempenho.append(f"=== {dataset} / {modelo} ===\n{info['bloco_desempenho']}".strip())
        if info["bloco_custo"]:
            agreg_custo.append(f"=== {dataset} / {modelo} ===\n{info['bloco_custo']}".strip())

        encontrados += 1

    if encontrados == 0:
        print(f"Nenhum resumo encontrado para dataset '{dataset_arg}' em {ESTAT_DIR}")
        sys.exit(1)

    # ===== Saída: dentro de estatisticas/<dataset>_custo =====
    OUT_DIR = ESTAT_DIR / f"{dataset_arg}_custo"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    out_tudo = OUT_DIR / f"{dataset_alvo}_tudo.txt"
    out_desempenho = OUT_DIR / f"{dataset_alvo}_desempenho.txt"
    out_custo = OUT_DIR / f"{dataset_alvo}_custo.txt"

    sep = "\n\n" + ("-" * 64) + "\n\n"
    out_tudo.write_text(sep.join(agreg_tudo) + "\n", encoding="utf-8")
    out_desempenho.write_text(sep.join(agreg_desempenho) + "\n", encoding="utf-8")
    out_custo.write_text(sep.join(agreg_custo) + "\n", encoding="utf-8")

    print("[OK] Gerados em:", OUT_DIR)
    print(" -", out_tudo.name)
    print(" -", out_desempenho.name)
    print(" -", out_custo.name)

if __name__ == "__main__":
    main()
