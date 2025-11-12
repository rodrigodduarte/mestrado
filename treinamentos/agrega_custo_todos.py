#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import sys

"""
Agrega todos os <dataset>_custo.txt em estatisticas/*_custo/ para um único
arquivo estatisticas/custos_todos.txt.

Uso:
  python agrega_todos_custos.py
  # opcional: definir nome de saída
  python agrega_todos_custos.py custos_agregado.txt
"""

def main():
    ROOT = Path.cwd()                 # rode em .../treinamentos
    ESTAT_DIR = ROOT / "estatisticas" # onde estão os diretórios *_custo

    if not ESTAT_DIR.is_dir():
        print(f"ERRO: diretório não encontrado: {ESTAT_DIR}", file=sys.stderr)
        sys.exit(2)

    # Nome do arquivo de saída (pode ser passado como 1º arg)
    out_name = sys.argv[1] if len(sys.argv) > 1 else "custos_todos.txt"
    OUT_FILE = ESTAT_DIR / out_name

    # Procurar diretórios que terminam com "_custo"
    custo_dirs = sorted([p for p in ESTAT_DIR.iterdir() if p.is_dir() and p.name.endswith("_custo")])

    agregados = []
    encontrados = 0

    for d in custo_dirs:
        # todos os *_custo.txt dentro do diretório
        for f in sorted(d.glob("*_custo.txt")):
            try:
                txt = f.read_text(encoding="utf-8", errors="replace").strip()
            except Exception as e:
                print(f"[AVISO] Não foi possível ler {f}: {e}", file=sys.stderr)
                continue

            # Tentar inferir o nome do dataset
            # 1) do nome do diretório (ex.: D2_custo -> dataset = D2)
            dataset_dir = d.name[:-6] if d.name.lower().endswith("_custo") else d.name
            # 2) ou do arquivo (ex.: d2_custo.txt -> dataset = d2)
            dataset_file = f.stem[:-6] if f.name.lower().endswith("_custo.txt") else f.stem

            # Preferir o nome do diretório (mantém caixa ex.: D2, swedish, flavia)
            dataset = dataset_dir if dataset_dir else dataset_file

            header = []
            header.append(f"=== DATASET: {dataset} ===")
            header.append(f"(fonte: {f})")
            bloco = "\n".join(header) + "\n\n" + txt
            agregados.append(bloco)
            encontrados += 1

    if encontrados == 0:
        print("Nenhum arquivo *_custo.txt encontrado em estatisticas/*_custo/", file=sys.stderr)
        sys.exit(1)

    sep = "\n\n" + ("-" * 72) + "\n\n"
    OUT_FILE.write_text(sep.join(agregados) + "\n", encoding="utf-8")

    print(f"[OK] Arquivo agregado gerado: {OUT_FILE}")
    print(f"[OK] Total de arquivos mesclados: {encontrados}")

if __name__ == "__main__":
    main()
