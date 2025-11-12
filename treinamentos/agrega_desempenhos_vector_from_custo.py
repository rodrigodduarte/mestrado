#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import re
import sys
from datetime import datetime

# cabeçalhos no formato: === <dataset> / <modelo> ===
HEADER_RE = re.compile(r"^===\s*(?P<dataset>.+?)\s*/\s*(?P<modelo>.+?)\s*===\s*$")

def split_blocks(text: str):
    """Divide o arquivo em blocos por cabeçalho '=== dataset / modelo ==='."""
    lines = text.splitlines()
    blocks = []
    cur_header = None
    cur_lines = []
    for ln in lines:
        m = HEADER_RE.match(ln.strip())
        if m:
            # fecha bloco anterior
            if cur_header is not None:
                blocks.append((cur_header, "\n".join(cur_lines).strip()))
            cur_header = (m.group("dataset").strip(), m.group("modelo").strip())
            cur_lines = [ln]  # mantém linha de cabeçalho
        else:
            if cur_header is not None:
                cur_lines.append(ln)
    if cur_header is not None:
        blocks.append((cur_header, "\n".join(cur_lines).strip()))
    return blocks

def main():
    TREINAMENTOS = Path.cwd()  # rode em .../treinamentos
    ESTAT = TREINAMENTOS / "estatisticas"
    if not ESTAT.is_dir():
        print(f"ERRO: não encontrei {ESTAT}", file=sys.stderr)
        sys.exit(2)

    out_name = sys.argv[1] if len(sys.argv) > 1 else "desempenhos_vector_todos.txt"
    OUT = ESTAT / out_name

    # evitar sobrescrita: se já existir, cria com timestamp
    if OUT.exists():
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        OUT = ESTAT / f"{OUT.stem}-{ts}{OUT.suffix}"

    agreg = []
    fontes = []
    count_blocks = 0

    # varre estatisticas/*_custo/*_desempenho.txt
    for d in sorted(p for p in ESTAT.iterdir() if p.is_dir() and p.name.endswith("_custo")):
        for f in sorted(d.glob("*_desempenho.txt")):
            try:
                txt = f.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                print(f"[AVISO] Falha lendo {f}: {e}", file=sys.stderr)
                continue

            for (dataset, modelo), bloco in split_blocks(txt):
                if "vector" in modelo.lower():
                    header = f"=== {dataset} / {modelo} ===\n(fonte: {f})\n"
                    agreg.append(header + "\n" + bloco)
                    fontes.append(str(f))
                    count_blocks += 1

    if count_blocks == 0:
        print("Nenhum bloco de desempenho com modelo '*vector' encontrado em estatisticas/*_custo/*_desempenho.txt", file=sys.stderr)
        sys.exit(1)

    sep = "\n\n" + ("-" * 72) + "\n\n"
    OUT.write_text(sep.join(agreg) + "\n", encoding="utf-8")

    # manifest opcional das fontes
    (ESTAT / "desempenhos_vector_manifest.txt").write_text("\n".join(sorted(set(fontes))) + "\n", encoding="utf-8")

    print(f"[OK] Gerado: {OUT}")
    print(f"[OK] Blocos (vector) mesclados: {count_blocks}")
    print(f"[OK] Manifest: {(ESTAT / 'desempenhos_vector_manifest.txt')}")

if __name__ == "__main__":
    main()
