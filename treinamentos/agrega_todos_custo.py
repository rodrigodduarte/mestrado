#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import sys

"""
Agrega todos os *_custo.txt em treinamentos/estatisticas/*_custo/
e salva em treinamentos/estatisticas/custos_todos.txt (ou nome escolhido).

Uso (recomendado):
  # rodando de QUALQUER lugar, apontando para o diretório 'treinamentos'
  python agrega_todos_custos.py /caminho/para/treinamentos

Uso (sem argumento):
  # script tenta achar 'treinamentos/estatisticas' a partir do diretório atual
  python agrega_todos_custos.py

Uso (com nome de saída opcional):
  python agrega_todos_custos.py /caminho/treinamentos custos_agregado.txt
"""

def find_estatisticas(base_treinamentos: Path) -> Path:
    est = base_treinamentos / "estatisticas"
    if not est.is_dir():
        raise FileNotFoundError(f"Diretório não encontrado: {est}")
    return est

def main():
    args = sys.argv[1:]
    if len(args) == 0:
        # tentar achar automaticamente: procurar 'treinamentos/estatisticas' subindo a partir do CWD
        cwd = Path.cwd()
        cand = None
        for p in [cwd] + list(cwd.parents):
            t = p / "treinamentos"
            if (t / "estatisticas").is_dir():
                cand = t
                break
        if cand is None:
            print("ERRO: Não encontrei 'treinamentos/estatisticas' a partir do diretório atual.\n"
                  "Dica: chame com o caminho de 'treinamentos', ex.: "
                  "python agrega_todos_custos.py ~/Documentos/projeto/treinamentos", file=sys.stderr)
            sys.exit(2)
        treinamentos_dir = cand
        out_name = "custos_todos.txt"
    elif len(args) == 1:
        treinamentos_dir = Path(args[0]).expanduser().resolve()
        out_name = "custos_todos.txt"
    else:
        treinamentos_dir = Path(args[0]).expanduser().resolve()
        out_name = args[1]

    try:
        ESTAT_DIR = find_estatisticas(treinamentos_dir)
    except FileNotFoundError as e:
        print(f"ERRO: {e}", file=sys.stderr)
        sys.exit(2)

    OUT_FILE = ESTAT_DIR / out_name

    # diretórios que terminam com "_custo" dentro de treinamentos/estatisticas
    custo_dirs = sorted([p for p in ESTAT_DIR.iterdir() if p.is_dir() and p.name.endswith("_custo")])

    agregados = []
    encontrados = 0

    for d in custo_dirs:
        for f in sorted(d.glob("*_custo.txt")):
            try:
                txt = f.read_text(encoding="utf-8", errors="replace").strip()
            except Exception as e:
                print(f"[AVISO] Não foi possível ler {f}: {e}", file=sys.stderr)
                continue

            dataset_dir = d.name[:-6] if d.name.lower().endswith("_custo") else d.name
            dataset_file = f.stem[:-6] if f.name.lower().endswith("_custo.txt") else f.stem
            dataset = dataset_dir or dataset_file

            header = f"=== DATASET: {dataset} ===\n(fonte: {f})\n"
            agregados.append(header + "\n" + txt)
            encontrados += 1

    if encontrados == 0:
        print("Nenhum arquivo *_custo.txt encontrado em treinamentos/estatisticas/*_custo/", file=sys.stderr)
        sys.exit(1)

    sep = "\n\n" + ("-" * 72) + "\n\n"
    OUT_FILE.write_text(sep.join(agregados) + "\n", encoding="utf-8")

    print(f"[OK] Arquivo agregado gerado: {OUT_FILE}")
    print(f"[OK] Total de arquivos mesclados: {encontrados}")

if __name__ == "__main__":
    main()
